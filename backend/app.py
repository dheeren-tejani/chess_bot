"""FastAPI backend — C++ UCI engine, GPU-batched, subprocess-backed.

API contract (unchanged from the pure-Python version):
    POST /api/move            {fen, moves, depth} -> {best_move, evaluation}
    POST /api/games            GamePayload         -> {replay_code}
    GET  /api/games/{code}                         -> GamePayload

The move endpoint now shells out to the compiled `engine uci` binary (the
same one used by self-play and matches), which batches leaves across 32
parallel searches and runs on the GPU. Expected latency: ~200-400 ms/move
at 1000 visits, vs 1-2 s for the old in-process Python MCTS.

`depth` in the request is still accepted for API compatibility but ignored.

Run from the repo root (needs the engine binary at engine/build/engine,
or set CHESS_ENGINE_BIN):
    uvicorn serve.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
import secrets
import shlex
import site
import string
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# -- configuration (env-overridable) -----------------------------------------
ENGINE_BIN = os.environ.get(
    "CHESS_ENGINE_BIN", str(REPO_ROOT / "engine" / "build" / "engine"))
MODEL_PATH = os.environ.get("CHESS_MODEL_PATH", "models/champion.onnx")
VISITS = int(os.environ.get("CHESS_VISITS", "4096"))
MOVETIME_MS = int(os.environ.get("CHESS_MOVETIME_MS", "0"))   # 0 => use visits
PREFER_GPU = os.environ.get("CHESS_DEVICE", "auto").lower() != "cpu"
ENGINE_TIMEOUT_S = float(os.environ.get("CHESS_ENGINE_TIMEOUT_S", "60"))
ENGINE_STARTUP_TIMEOUT_S = float(os.environ.get("CHESS_ENGINE_STARTUP_S", "180"))

DATA_DIR = Path(os.environ.get(
    "CHESS_DATA_DIR", Path(__file__).resolve().parent / "data"))
GAMES_DIR = DATA_DIR / "games"
GAMES_DIR.mkdir(parents=True, exist_ok=True)
CORS_ORIGINS = os.environ.get("CHESS_CORS_ORIGINS", "*").split(",")

app = FastAPI(title="chess-bot serving backend (cpp-uci)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- engine client -----------------------------------------------------------
class EngineError(RuntimeError):
    pass


def _engine_env() -> dict:
    """Inherit the current env; auto-inject LD_LIBRARY_PATH from torch's
    bundled nvidia libs so ORT's CUDA provider actually loads. Mirrors what
    trainer/orchestrator.py does for the self-play subprocess."""
    env = os.environ.copy()
    if "nvidia" in env.get("LD_LIBRARY_PATH", ""):
        return env
    try:
        paths = []
        for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
            nvidia = Path(sp) / "nvidia"
            if nvidia.exists():
                paths.extend(str(p) for p in sorted(nvidia.glob("*/lib")))
        if paths:
            env["LD_LIBRARY_PATH"] = (
                ":".join(paths) + ":" + env.get("LD_LIBRARY_PATH", ""))
            print(f"[engine] injected LD_LIBRARY_PATH ({len(paths)} dirs)",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[engine] LD_LIBRARY_PATH auto-inject skipped: {e}", flush=True)
    return env


class EngineClient:
    """Persistent C++ UCI engine subprocess.

    Thread-safe: FastAPI runs sync endpoints in a thread pool, and UCI is a
    serial protocol, so all IO is guarded by a lock. Each search is
    ~200-400 ms at 1000 visits; contention is acceptable for a personal
    server. If you need higher throughput, run multiple server workers with
    separate ports, not more threads here.
    """

    def __init__(self, engine_bin: str, model_path: str, visits: int,
                 prefer_gpu: bool, movetime_ms: int = 0):
        self.engine_bin = engine_bin
        self.model_path = model_path
        self.visits = visits
        self.movetime_ms = movetime_ms
        self.prefer_gpu = prefer_gpu

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_tail: list[str] = []
        self._spawn()

    def _spawn(self):
        model_abs = self.model_path
        if not os.path.isabs(model_abs):
            model_abs = str((REPO_ROOT / model_abs).resolve())

        cmd = [self.engine_bin, "uci", "--model", model_abs,
               "--visits", str(self.visits)]
        if not self.prefer_gpu:
            cmd.append("--cpu")

        print(f"[engine] spawning: {shlex.join(cmd)}", flush=True)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            env=_engine_env(),
        )

        # Drain stderr on a background thread and keep a tail for error
        # reporting. Critical for surfacing CUDA-load failures.
        self._stderr_tail = []
        proc = self._proc

        def pump_err():
            assert proc.stderr is not None
            for line in proc.stderr:
                self._stderr_tail.append(line.rstrip())
                del self._stderr_tail[:-40]
                sys.stderr.write(f"[engine] {line}")
                sys.stderr.flush()

        threading.Thread(target=pump_err, daemon=True).start()

        # Handshake. The first `isready` blocks on ONNX load + warmup.
        self._send("uci")
        self._read_until("uciok", ENGINE_STARTUP_TIMEOUT_S)
        self._send("isready")
        self._read_until("readyok", ENGINE_STARTUP_TIMEOUT_S)
        print("[engine] ready", flush=True)

    def _send(self, cmd: str):
        if self._proc is None or self._proc.poll() is not None:
            raise EngineError("engine subprocess not alive")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(cmd + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise EngineError(f"engine pipe broken: {e}") from e

    def _read_until(self, prefix: str, timeout_s: float) -> list[str]:
        if self._proc is None or self._proc.stdout is None:
            raise EngineError("engine subprocess not alive")
        lines: list[str] = []
        deadline = time.monotonic() + timeout_s
        while True:
            if time.monotonic() >= deadline:
                tail = self._stderr_tail[-5:]
                raise EngineError(
                    f"timeout waiting for {prefix!r}; stderr tail: {tail}")
            line = self._proc.stdout.readline()
            if not line:
                tail = self._stderr_tail[-5:]
                raise EngineError(
                    f"engine EOF waiting for {prefix!r}; stderr tail: {tail}")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def get_move(self, position_cmd: str) -> tuple[str, int, int]:
        """Returns (best_move_uci, eval_cp_stm_relative, elapsed_ms).

        `position_cmd` must be a complete UCI `position ...` command.
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                print("[engine] process dead; respawning", flush=True)
                self._spawn()

            self._send(position_cmd)

            go_cmd = (f"go movetime {self.movetime_ms}"
                      if self.movetime_ms > 0
                      else f"go nodes {self.visits}")

            t0 = time.monotonic()
            self._send(go_cmd)
            lines = self._read_until("bestmove", ENGINE_TIMEOUT_S)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            best_move = "0000"
            eval_cp = 0
            for line in lines:
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]
                elif line.startswith("info ") and " score cp " in line:
                    parts = line.split()
                    try:
                        eval_cp = int(parts[parts.index("cp") + 1])
                    except (ValueError, IndexError):
                        pass

            if best_move == "0000":
                raise EngineError("engine returned no best move")
            return best_move, eval_cp, elapsed_ms

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:  # noqa: BLE001
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


engine: Optional[EngineClient] = None


@app.on_event("startup")
def _startup():
    global engine
    model_abs = Path(MODEL_PATH)
    if not model_abs.is_absolute():
        model_abs = (REPO_ROOT / model_abs).resolve()
    if not model_abs.exists():
        raise RuntimeError(
            f"model not found at {model_abs} - set CHESS_MODEL_PATH, or "
            f"export one with `python trainer/export_onnx.py --checkpoint "
            f"<ckpt> --out {model_abs}`")

    engine_bin = Path(ENGINE_BIN)
    if not engine_bin.exists():
        raise RuntimeError(
            f"engine binary not found at {engine_bin} - build it with "
            f"`cd engine && cmake --build build -j`, or set CHESS_ENGINE_BIN")

    engine = EngineClient(
        engine_bin=str(engine_bin),
        model_path=str(model_abs),
        visits=VISITS,
        prefer_gpu=PREFER_GPU,
        movetime_ms=MOVETIME_MS,
    )
    print(f"[serve] ready | backend=cpp-uci visits={VISITS} "
          f"movetime_ms={MOVETIME_MS} gpu_pref={PREFER_GPU}", flush=True)


@app.on_event("shutdown")
def _shutdown():
    if engine is not None:
        engine.close()


# -- request/response schemas (unchanged) ------------------------------------
class MoveRequest(BaseModel):
    fen: str
    moves: list[str] = []
    depth: Optional[int] = None   # accepted for compatibility, unused


class MoveResponse(BaseModel):
    best_move: str
    evaluation: int


class GamePayloadIn(BaseModel):
    moves: list[str]
    result: Optional[str] = None
    evaluations: Optional[list[float]] = None
    player_color: Optional[str] = None
    started_at: Optional[int] = None
    duration_ms: Optional[int] = None
    move_times_ms: Optional[list[int]] = None


class StoredGame(BaseModel):
    code: str
    created_at: str
    moves: list[str]
    result: Optional[str] = None
    winner: Optional[str] = None
    ply_count: int
    evaluations: Optional[list[float]] = None
    final_eval: Optional[float] = None
    final_fen: str
    player_color: str = "white"
    started_at: Optional[int] = None
    duration_ms: Optional[int] = None
    move_times_ms: Optional[list[int]] = None
    engine: dict


class GameSummary(BaseModel):
    code: str
    created_at: str
    result: Optional[str] = None
    ply_count: int
    duration_ms: Optional[int] = None


class SaveGameResponse(BaseModel):
    replay_code: str


# -- position reconstruction -------------------------------------------------
def _build_position(moves: list[str], fen: str) -> tuple[str, chess.Board]:
    """Return (uci_position_command, board_for_validation).

    Prefer replaying `moves` from startpos: that preserves full game history
    for the engine's encoder (8-ply planes + repetition tracking). Falls back
    to `position fen <fen>` when the move list is missing or illegal - the
    history planes will then be blank, which the encoder tolerates.
    """
    if moves:
        board = chess.Board()
        uci_moves: list[str] = []
        try:
            for i, san in enumerate(moves):
                m = board.parse_san(san)
                uci_moves.append(m.uci())
                board.push(m)
        except Exception as exc:  # noqa: BLE001
            print(f"[serve] WARN replay failed at ply {len(uci_moves)+1} "
                  f"({exc}); falling back to FEN", flush=True)
        else:
            if fen:
                try:
                    client_board = chess.Board(fen)
                    if client_board.board_fen() != board.board_fen():
                        print(f"[serve] WARN fen mismatch: "
                              f"replay={board.board_fen()} "
                              f"client={client_board.board_fen()}", flush=True)
                except Exception:
                    pass
            return ("position startpos moves " + " ".join(uci_moves), board)

    try:
        board = chess.Board(fen)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"invalid FEN {fen!r}: {exc}") from exc
    return (f"position fen {fen}", board)


# -- routes ------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": engine is not None and engine.alive(),
        "backend": "cpp-uci",
        "engine_bin": ENGINE_BIN,
        "model": MODEL_PATH,
        "visits": VISITS,
        "movetime_ms": MOVETIME_MS,
    }


@app.post("/api/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    if engine is None:
        raise HTTPException(503, "engine not initialized")

    position_cmd, board = _build_position(req.moves, req.fen)
    if board.is_game_over():
        raise HTTPException(400, "position is already game over")

    try:
        best_move, eval_cp, elapsed_ms = engine.get_move(position_cmd)
    except EngineError as exc:
        raise HTTPException(500, f"engine error: {exc}") from exc

    # Engine reports cp from the SIDE-TO-MOVE's perspective. The old pure-
    # Python serve returned cp from WHITE's perspective; preserve that contract.
    if board.turn == chess.BLACK:
        eval_cp = -eval_cp
    eval_cp = max(-1500, min(1500, eval_cp))

    print(f"[serve] move ply={len(req.moves)+1} visits={VISITS} "
          f"{elapsed_ms}ms -> {best_move} eval={eval_cp}", flush=True)
    return MoveResponse(best_move=best_move, evaluation=eval_cp)


# -- game storage (unchanged) ------------------------------------------------
CODE_ALPHABET = string.ascii_uppercase + string.digits


def _game_path(code: str) -> Path:
    return GAMES_DIR / f"{code}.json"


def _new_replay_code() -> str:
    for _ in range(50):
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        if not _game_path(code).exists():
            return code
    raise HTTPException(500, "could not allocate a unique replay code, try again")


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _validate_and_finalize_moves(sans: list[str]) -> chess.Board:
    board = chess.Board()
    for i, san in enumerate(sans):
        try:
            board.push_san(san)
        except Exception as exc:
            raise HTTPException(
                400, f"invalid move at ply {i + 1} ({san!r}): {exc}") from exc
    return board


@app.post("/api/games", response_model=SaveGameResponse)
def save_game(payload: GamePayloadIn):
    if not payload.moves:
        raise HTTPException(400, "no moves to save")

    final_board = _validate_and_finalize_moves(payload.moves)

    result = payload.result
    winner = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}.get(result or "")

    duration_ms = payload.duration_ms
    if duration_ms is None and payload.started_at is not None:
        duration_ms = int(time.time() * 1000) - payload.started_at

    code = _new_replay_code()
    record = StoredGame(
        code=code,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        moves=payload.moves,
        result=result,
        winner=winner,
        ply_count=len(payload.moves),
        evaluations=payload.evaluations,
        player_color=(payload.player_color or "white").lower(),
        final_eval=(payload.evaluations[-1] if payload.evaluations else None),
        final_fen=final_board.fen(),
        started_at=payload.started_at,
        duration_ms=duration_ms,
        move_times_ms=payload.move_times_ms,
        engine={"backend": "cpp-uci", "visits": VISITS,
                "movetime_ms": MOVETIME_MS, "model": str(MODEL_PATH)},
    )
    _atomic_write_json(_game_path(code), record.model_dump())
    return SaveGameResponse(replay_code=code)


@app.get("/api/games/{code}", response_model=StoredGame)
def load_game(code: str):
    path = _game_path(code.strip().upper())
    if not path.exists():
        raise HTTPException(404, "not found")
    return json.loads(path.read_text())


@app.get("/api/games", response_model=list[GameSummary])
def list_games(limit: int = 50):
    files = sorted(GAMES_DIR.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:limit]:
        try:
            g = json.loads(f.read_text())
            out.append(GameSummary(
                code=g["code"], created_at=g["created_at"],
                result=g.get("result"), ply_count=g["ply_count"],
                duration_ms=g.get("duration_ms")))
        except Exception:
            continue
    return out