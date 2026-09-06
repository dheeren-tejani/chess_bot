"""ONNX Runtime evaluator mirroring engine/src/nn_client.cpp (bit-packed uint8
planes input) plus the policy/value post-processing from
engine/src/selfplay.cpp's EvalBatcher::fill_task (subset-softmax over legal
moves, WDL->scalar value with contempt blending).
"""
from __future__ import annotations

import threading
from typing import Sequence

import chess
import numpy as np
import onnxruntime as ort

from trainer.encoding import NUM_PLANES, SCALAR_COUNT, encode_board


class OnnxEvaluator:
    def __init__(self, model_path: str, prefer_gpu: bool = True):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.is_gpu = providers[0] == "CUDAExecutionProvider"

        # onnxruntime sessions are generally safe for concurrent Run() calls,
        # but we serialize anyway - this is a small local dev server, not a
        # high-throughput service, and it keeps behaviour simple/predictable.
        self._lock = threading.Lock()

    def run_batch(self, boards: Sequence[chess.Board], histories: Sequence[list]):
        """Returns (policy[N,4672], wdl[N,3], material[N]) raw logits/values."""
        n = len(boards)
        planes_u8 = np.empty((n, NUM_PLANES, 8), dtype=np.uint8)
        scalars = np.empty((n, SCALAR_COUNT), dtype=np.float32)
        for i, (b, h) in enumerate(zip(boards, histories)):
            planes, sc = encode_board(b, h)
            # Same reinterpretation as engine/src/nn_client.cpp: a uint64 word
            # per plane, viewed as 8 little-endian bytes (one per rank).
            planes_u8[i] = planes.view(np.uint8).reshape(NUM_PLANES, 8)
            scalars[i] = sc

        with self._lock:
            policy, wdl, material = self.session.run(
                ["policy", "wdl", "material"],
                {"planes": planes_u8, "scalars": scalars},
            )
        return policy, wdl, material.reshape(-1)


def priors_and_value(policy_row: np.ndarray, wdl_row: np.ndarray, material_row: float,
                      policy_idx: list[int], stm: bool, root_stm: bool,
                      contempt: float, value_material_alpha: float) -> tuple[list[float], float]:
    """Mirrors EvalBatcher::fill_task in engine/src/selfplay.cpp."""
    logits = policy_row[policy_idx]
    logits = logits - logits.max()
    ex = np.exp(logits)
    priors = (ex / ex.sum()).tolist()

    w = wdl_row - wdl_row.max()
    ew = np.exp(w)
    e0, e1, e2 = float(ew[0]), float(ew[1]), float(ew[2])
    es = e0 + e1 + e2

    c_eff = contempt if stm == root_stm else (1.0 - contempt)
    base = (e0 + c_eff * e1) / es
    value = 2.0 * base - 1.0

    if value_material_alpha != 0.0:
        # train.py trains the material head on target = material/8, matching
        # the C++ comment in selfplay.cpp - don't divide again here.
        value += value_material_alpha * float(material_row)

    value = max(-1.0, min(1.0, value))
    return priors, value