"""Python port of engine/src/mcts.cpp for serving a trained checkpoint.

This mirrors the C++ `Search` class closely enough to be a faithful port:
  - PUCT selection with the same cpuct/FPU formulas (mcts.cpp Search::select_child)
  - virtual loss during batched leaf gathering (mcts.cpp Search::backprop /
    compensate_vl), so a single-threaded search can still gather a full batch
    of diverse leaves per round before calling the network
  - search-internal twofold-repetition + 50-move pruning
    (mcts.cpp Search::descend_once)

Deliberately DOES NOT port:
  - root Dirichlet noise (self-play exploration only)
  - prior-ply / temperature move sampling (self-play exploration only)
  - tree reuse across plies (`keep_subtree`) - each HTTP request is a fresh
    search from scratch, since the frontend is stateless per move

Move selection here is always "greedy": the root edge with the most visits
(ties broken by prior), i.e. what engine/src/selfplay.cpp's `pick_move` does
once past all the exploration windows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import chess

from trainer.encoding import _pos_key, move_policy_index

ST_UNEXPANDED, ST_INFLIGHT, ST_EXPANDED, ST_TERMINAL = 0, 1, 2, 3


class Edge:
    __slots__ = ("move", "prior", "policy_idx", "visits", "value_sum", "child")

    def __init__(self, move: chess.Move, prior: float, policy_idx: int):
        self.move = move
        self.prior = prior
        self.policy_idx = policy_idx
        self.visits = 0
        self.value_sum = 0.0
        self.child: Optional[int] = None


class Node:
    __slots__ = ("state", "term_value", "edges")

    def __init__(self):
        self.state = ST_UNEXPANDED
        self.term_value = 0.0
        self.edges: list[Edge] = []


@dataclass
class EvalTask:
    node: int
    path: list[tuple[int, int]]
    board: chess.Board             # leaf position (board.turn == stm)
    legal: list[chess.Move]
    policy_idx: list[int]
    stm: bool                      # chess.WHITE / chess.BLACK
    root_stm: bool
    history: list[chess.Board]     # game history + search walk, ends with `board`


class Search:
    CPUCT_INIT = 1.25
    CPUCT_BASE = 19652.0
    FPU_REDUCTION = 0.2
    VIRTUAL_LOSS = 1.0

    def __init__(self, root_board: chess.Board, game_history: list[chess.Board],
                 use_twofold_draw: bool = True):
        """`game_history` must end with a board equal to `root_board` (same
        convention as the engine's `hist_`: full game so far, root position last)."""
        self.root_board = root_board.copy(stack=False)
        self.history = game_history
        self.use_twofold_draw = use_twofold_draw
        self.nodes: list[Node] = [Node()]
        self.root = 0

    # -- small helpers ---------------------------------------------------
    def _cpuct(self, parent_visits: float) -> float:
        return self.CPUCT_INIT + math.log(
            (parent_visits + self.CPUCT_BASE + 1.0) / self.CPUCT_BASE)

    def _draw_value_for(self, stm: bool, contempt: float) -> float:
        # Search-side draw value with contempt, mirroring the blend used for
        # real WDL predictions in onnx_model.priors_and_value (a rule-based
        # draw is just a pure-draw WDL distribution run through the same
        # formula). contempt=0.5 (the default here) is neutral / no bias.
        c_eff = contempt if stm == self.root_board.turn else (1.0 - contempt)
        return 2.0 * c_eff - 1.0

    def _new_node(self) -> int:
        self.nodes.append(Node())
        return len(self.nodes) - 1

    @staticmethod
    def _repeats(boards: list[chess.Board], board: chess.Board) -> int:
        key = _pos_key(board)
        return sum(1 for b in boards if _pos_key(b) == key)

    def root_visits(self) -> int:
        return sum(e.visits for e in self.nodes[self.root].edges)

    def root_q(self) -> float:
        """Average backed-up value from the root's side-to-move perspective."""
        edges = self.nodes[self.root].edges
        v = sum(e.visits for e in edges)
        if v == 0:
            return 0.0
        s = sum(e.value_sum for e in edges)
        return s / v

    def pick_move_greedy(self) -> Edge:
        edges = self.nodes[self.root].edges
        return max(edges, key=lambda e: (e.visits, e.prior))

    # -- PUCT selection ----------------------------------------------------
    def _select_child(self, node_idx: int, path: list[tuple[int, int]]) -> int:
        n = self.nodes[node_idx]
        edges = n.edges
        total_v = sum(e.visits for e in edges)
        total_s = sum(e.value_sum for e in edges)
        parent_q = (total_s / total_v) if total_v else 0.0
        fpu = parent_q - self.FPU_REDUCTION
        sqrt_n = math.sqrt(max(total_v, 1))
        c = self._cpuct(total_v)

        best, best_score = -1, -1e30
        for i, e in enumerate(edges):
            q = (e.value_sum / e.visits) if e.visits else fpu
            u = c * e.prior * sqrt_n / (1.0 + e.visits)
            score = q + u
            if score > best_score:
                best_score, best = score, i
        path.append((node_idx, best))
        return best

    def _backprop(self, path: list[tuple[int, int]], value_from_leaf_stm: float) -> None:
        contrib = -value_from_leaf_stm
        for node_idx, ei in reversed(path):
            e = self.nodes[node_idx].edges[ei]
            e.visits += 1
            e.value_sum += contrib + self.VIRTUAL_LOSS
            contrib = -contrib

    def _compensate_vl(self, path: list[tuple[int, int]]) -> None:
        for node_idx, ei in reversed(path):
            self.nodes[node_idx].edges[ei].value_sum += self.VIRTUAL_LOSS

    # -- one simulation ------------------------------------------------------
    def descend_once(self, contempt: float) -> Optional[EvalTask]:
        path: list[tuple[int, int]] = []
        walk: list[chess.Board] = []
        node = self.root

        def cur() -> chess.Board:
            return walk[-1] if walk else self.root_board

        while True:
            n = self.nodes[node]

            if n.state == ST_TERMINAL:
                self._backprop(path, n.term_value)
                return None
            if n.state == ST_INFLIGHT:
                self._compensate_vl(path)
                return None
            if n.state == ST_UNEXPANDED:
                board = cur()
                legal = list(board.legal_moves)
                if not legal:
                    term_value = -1.0 if board.is_check() else self._draw_value_for(board.turn, contempt)
                    n.state, n.term_value = ST_TERMINAL, term_value
                    self._backprop(path, term_value)
                    return None
                n.state = ST_INFLIGHT
                policy_idx = [move_policy_index(mv, board) for mv in legal]
                return EvalTask(
                    node=node, path=list(path), board=board, legal=legal,
                    policy_idx=policy_idx, stm=board.turn, root_stm=self.root_board.turn,
                    history=self.history + walk)

            ei = self._select_child(node, path)
            edge = n.edges[ei]
            edge.value_sum -= self.VIRTUAL_LOSS
            if edge.child is None:
                edge.child = self._new_node()

            nb = cur().copy(stack=False)
            nb.push(edge.move)

            if self.use_twofold_draw:
                # "seen before" set = full game history (incl. root) + this
                # descent's walk so far, NOT including `nb` itself.
                seen = self._repeats(self.history + walk, nb)
                if seen >= 1:
                    cn = self.nodes[edge.child]
                    cn.state = ST_TERMINAL
                    cn.term_value = self._draw_value_for(nb.turn, contempt)
                    self._backprop(path, cn.term_value)
                    return None
            if nb.halfmove_clock >= 100:
                cn = self.nodes[edge.child]
                cn.state = ST_TERMINAL
                cn.term_value = self._draw_value_for(nb.turn, contempt)
                self._backprop(path, cn.term_value)
                return None

            walk.append(nb)
            node = edge.child

    def gather_round(self, max_leaves: int, contempt: float) -> list[EvalTask]:
        out: list[EvalTask] = []
        for _ in range(max_leaves):
            t = self.descend_once(contempt)
            if t is not None:
                out.append(t)
        return out

    def complete_task(self, task: EvalTask, priors: list[float], value: float) -> None:
        n = self.nodes[task.node]
        if n.state != ST_INFLIGHT:
            return
        n.edges = [Edge(mv, p, pidx)
                   for mv, p, pidx in zip(task.legal, priors, task.policy_idx)]
        n.state = ST_EXPANDED
        self._backprop(task.path, value)