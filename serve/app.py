"""FastAPI backend for playing/testing a self-play checkpoint from the browser.

Implements exactly the contract expected by the frontend's `api.ts`:
    POST /api/move            {fen, moves, depth} -> {best_move, evaluation}
    POST /api/games            GamePayload         -> {replay_code}
    GET  /api/games/{code}                         -> GamePayload

Search is always run for CHESS_VISITS (default 1024) MCTS visits and the move
is chosen GREEDILY (most-visited root edge) - no temperature, no Dirichlet
noise, no tree reuse across requests. `depth` in the request is accepted for
API compatibility but ignored.

Run from the repo root (needs `trainer/` importable):
    uvicorn serve.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import math
import os
import secrets
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from serve.mcts import ST_TERMINAL, Search           # noqa: E402
from serve.onnx_model import OnnxEvaluator, priors_and_value  # noqa: E402

# -- configuration (env-overridable) -----------------------------------------
MODEL_PATH = os.environ.get("CHESS_MODEL_PATH", "models/champion.onnx")
VISITS = int(os.environ.get("CHESS_VISITS", "1024"))
LEAVES_PER_ROUND = int(os.environ.get("CHESS_LEAVES_PER_ROUND", "32"))
CONTEMPT = float(os.environ.get("CHESS_CONTEMPT", "0.5"))          # 0.5 = neutral, no contempt
VALUE_MATERIAL_ALPHA = float(os.environ.get("CHESS_VALUE_MATERIAL_ALPHA", "0.0"))
CP_LOGISTIC_SCALE = float(os.environ.get("CHESS_CP_SCALE", "400.0"))
PREFER_GPU = os.environ.get("CHESS_DEVICE", "auto").lower() != "cpu"
DATA_DIR = Path(os.environ.get("CHESS_DATA_DIR", Path(__file__).resolve().parent / "data"))
GAMES_DIR = DATA_DIR / "games"
GAMES_DIR.mkdir(parents=True, exist_ok=True)
CORS_ORIGINS = os.environ.get("CHESS_CORS_ORIGINS", "*").split(",")

app = FastAPI(title="chess-bot serving backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator: Optional[OnnxEvaluator] = None


@app.on_event("startup")
def _load_model():
    global evaluator
    path = Path(MODEL_PATH)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise RuntimeError(
            f"model not found at {path} - set CHESS_MODEL_PATH, or export one with "
            f"`python trainer/export_onnx.py --checkpoint <ckpt> --out {path} "
            f"--blocks <N> --channels <N>`")
    evaluator = OnnxEvaluator(str(path), prefer_gpu=PREFER_GPU)
    print(f"[serve] loaded {path} (gpu={evaluator.is_gpu}) "
          f"visits={VISITS} mode=greedy contempt={CONTEMPT}")


# -- request/response schemas (mirrors frontend's GamePayload / api.ts) ------
class MoveRequest(BaseModel):
    fen: str
    moves: list[str] = []
    depth: Optional[int] = None   # accepted for compatibility, unused


class MoveResponse(BaseModel):
    best_move: str
    evaluation: int


class GamePayloadIn(BaseModel):
    """What the frontend sends when a game finishes. `duration_ms` /
    `move_times_ms` / `started_at` are optional so this stays compatible even
    if the frontend isn't sending timing data yet."""
    moves: list[str]
    result: Optional[str] = None
    evaluations: Optional[list[float]] = None
    started_at: Optional[int] = None        # epoch ms when the game began
    duration_ms: Optional[int] = None        # total elapsed play time, ms
    move_times_ms: Optional[list[int]] = None  # per-ply elapsed time since the previous move, ms


class StoredGame(BaseModel):
    """What we persist and hand back on replay - a superset of GamePayloadIn.
    The frontend's replay viewer only reads `moves` / `result` / `evaluations`,
    so the extra fields are just along for the ride (and useful for anything
    you build later - a game list, a duration column, etc)."""
    code: str
    created_at: str
    moves: list[str]
    result: Optional[str] = None
    winner: Optional[str] = None            # "white" | "black" | "draw" | None
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


# -- helpers ------------------------------------------------------------------
def _rebuild_position(fen: str, sans: list[str]) -> tuple[chess.Board, list[chess.Board]]:
    """Replay `sans` from the start position to reconstruct full game history
    (needed for the 8-ply history planes and repetition counting - see
    encoder.h). Falls back to the bare `fen` with no history if replay fails
    or ends up somewhere other than the given `fen`."""
    board = chess.Board()
    history: list[chess.Board] = [board.copy(stack=False)]
    ok = True
    for san in sans:
        try:
            board.push_san(san)
        except Exception:
            ok = False
            break
        history.append(board.copy(stack=False))
    if not ok or board.board_fen() != chess.Board(fen).board_fen():
        board = chess.Board(fen)
        history = [board.copy(stack=False)]
    return board, history


def _cp_from_value(value_root_stm: float, root_is_white: bool) -> int:
    """Convert a searched value in [-1, 1] (root_q(): the WDL-blended expected
    score for the root's side to move - i.e. win_prob + 0.5*draw_prob) into a
    centipawn number, using the standard logistic win-probability<->cp model
    (cp = scale * log10(p / (1-p)), scale=400 by default) rather than a plain
    linear scale. This is the same functional form engines use to report an
    eval: near 50/50 a small edge stays small, and it stretches out (instead
    of saturating) as a position becomes clearly winning."""
    value_white = value_root_stm if root_is_white else -value_root_stm
    p = (value_white + 1.0) / 2.0          # expected score for White, in (0, 1)
    eps = 1e-4
    p = min(1.0 - eps, max(eps, p))
    cp = CP_LOGISTIC_SCALE * math.log10(p / (1.0 - p))
    return max(-1500, min(1500, round(cp)))


def _run_search(board: chess.Board, history: list[chess.Board]) -> Search:
    search = Search(board, history, use_twofold_draw=True)
    while search.root_visits() < VISITS:
        if search.nodes[search.root].state == ST_TERMINAL:
            break  # root itself has no legal moves (mate/stalemate)
        tasks = search.gather_round(LEAVES_PER_ROUND, CONTEMPT)
        if not tasks:
            break
        boards = [t.board for t in tasks]
        histories = [t.history for t in tasks]
        policy, wdl, material = evaluator.run_batch(boards, histories)
        for row, task in enumerate(tasks):
            priors, value = priors_and_value(
                policy[row], wdl[row], material[row], task.policy_idx,
                task.stm, task.root_stm, CONTEMPT, VALUE_MATERIAL_ALPHA)
            search.complete_task(task, priors, value)
    return search


# -- routes ---------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": evaluator is not None,
            "gpu": evaluator.is_gpu if evaluator else None,
            "visits": VISITS, "mode": "greedy", "contempt": CONTEMPT}


@app.post("/api/move", response_model=MoveResponse)
def get_move(req: MoveRequest):
    if evaluator is None:
        raise HTTPException(503, "model not loaded")

    board, history = _rebuild_position(req.fen, req.moves)
    if board.is_game_over():
        raise HTTPException(400, "position is already game over")

    t0 = time.time()
    search = _run_search(board, history)
    elapsed_ms = (time.time() - t0) * 1000.0

    root_edges = search.nodes[search.root].edges
    if not root_edges:
        raise HTTPException(400, "no legal moves")

    best = search.pick_move_greedy()
    cp = _cp_from_value(search.root_q(), board.turn == chess.WHITE)
    print(f"[serve] move ply={len(req.moves)+1} visits={search.root_visits()} "
          f"{elapsed_ms:.0f}ms -> {best.move.uci()} eval={cp}")
    return MoveResponse(best_move=best.move.uci(), evaluation=cp)


CODE_ALPHABET = string.ascii_uppercase + string.digits  # 36^8 ≈ 2.8e12 combinations


def _game_path(code: str) -> Path:
    return GAMES_DIR / f"{code}.json"


def _new_replay_code() -> str:
    """Cryptographically random 8-char code, re-rolled on the (astronomically
    unlikely) chance of a collision with an existing saved game."""
    for _ in range(50):
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        if not _game_path(code).exists():
            return code
    # 50 collisions in a row against a 2.8e12-combination space essentially
    # never happens organically - if it does, something is actually wrong.
    raise HTTPException(500, "could not allocate a unique replay code, try again")


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write-to-temp-then-rename so a save is never left half-written on disk,
    even if the process is killed mid-write. `os.replace` is atomic on both
    POSIX and Windows."""
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _validate_and_finalize_moves(sans: list[str]) -> chess.Board:
    """Replays the move list from the start position so a corrupt/incomplete
    move list is rejected at save time instead of silently persisted -
    this is the "100% saved" guarantee: if it saved, it's a legal game."""
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
        duration_ms = int(time.time() * 1000) - payload.started_at  # best-effort fallback

    code = _new_replay_code()
    record = StoredGame(
        code=code,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        moves=payload.moves,
        result=result,
        winner=winner,
        ply_count=len(payload.moves),
        evaluations=payload.evaluations,
        final_eval=(payload.evaluations[-1] if payload.evaluations else None),
        final_fen=final_board.fen(),
        started_at=payload.started_at,
        duration_ms=duration_ms,
        move_times_ms=payload.move_times_ms,
        engine={"visits": VISITS, "mode": "greedy", "contempt": CONTEMPT,
                "model": str(MODEL_PATH)},
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
    """Bonus: not used by the current frontend, but handy for building a
    'recent games' browser later without needing a real database."""
    files = sorted(GAMES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for f in files[:limit]:
        try:
            g = json.loads(f.read_text())
            out.append(GameSummary(code=g["code"], created_at=g["created_at"],
                                    result=g.get("result"), ply_count=g["ply_count"],
                                    duration_ms=g.get("duration_ms")))
        except Exception:
            continue  # skip anything unreadable rather than 500 the whole list
    return out