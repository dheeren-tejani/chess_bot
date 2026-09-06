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
import os
import secrets
import string
import sys
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


class GamePayload(BaseModel):
    moves: list[str]
    result: Optional[str] = None
    evaluations: Optional[list[float]] = None


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
    """Rough linear WDL-value -> centipawn mapping for the frontend's eval
    bar. `value_root_stm` is root_q() (search-averaged, root side-to-move
    perspective, range [-1, 1]). Tune the 600 scale factor to taste."""
    value_white = value_root_stm if root_is_white else -value_root_stm
    cp = 600.0 * value_white
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

    search = _run_search(board, history)
    root_edges = search.nodes[search.root].edges
    if not root_edges:
        raise HTTPException(400, "no legal moves")

    best = search.pick_move_greedy()
    cp = _cp_from_value(search.root_q(), board.turn == chess.WHITE)
    return MoveResponse(best_move=best.move.uci(), evaluation=cp)


def _new_replay_code() -> str:
    alphabet = string.ascii_uppercase + string.digits
    for _ in range(20):
        code = "".join(secrets.choice(alphabet) for _ in range(8))
        if not (GAMES_DIR / f"{code}.json").exists():
            return code
    raise RuntimeError("could not allocate a unique replay code")


@app.post("/api/games", response_model=SaveGameResponse)
def save_game(payload: GamePayload):
    code = _new_replay_code()
    (GAMES_DIR / f"{code}.json").write_text(json.dumps(payload.model_dump()))
    return SaveGameResponse(replay_code=code)


@app.get("/api/games/{code}", response_model=GamePayload)
def load_game(code: str):
    path = GAMES_DIR / f"{code.strip().upper()}.json"
    if not path.exists():
        raise HTTPException(404, "not found")
    return json.loads(path.read_text())