"""
Modal deployment for the chess bot backend.

Base image: NVIDIA CUDA runtime (bundles cuDNN; Debian bookworm repos don't
have libcudnn8). ONNX Runtime C++ headers/libs are fetched from the official
release tarball, since libonnxruntime-dev isn't in bookworm.

Deploy:
    modal deploy modal_app.py
"""
from __future__ import annotations

import modal

# ============================================================
# 1. App + persistent volume
# ============================================================
app = modal.App("chess-bot")

games_volume = modal.Volume.from_name("chess-bot-games", create_if_missing=True)
chess_secrets = modal.Secret.from_dotenv()

# ============================================================
# 2. Image: CUDA runtime base + ORT C++ libs + Python deps
# ============================================================
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "cmake", "g++", "make", "pkg-config",
        "zlib1g-dev", "wget", "ca-certificates",
    )
    # ONNX Runtime C++ headers + shared library. The engine's CMakeLists.txt
    # looks for these at ../../third_party/onnxruntime/ (relative to build/),
    # so we install them exactly there. Also registered with ldconfig so the
    # runtime linker can find libonnxruntime.so.1 when the engine runs.
    .run_commands(
        "mkdir -p /root/third_party && "
        "cd /tmp && "
        "wget -q https://github.com/microsoft/onnxruntime/releases/download/"
        "v1.20.1/onnxruntime-linux-x64-1.20.1.tgz && "
        "tar -xzf onnxruntime-linux-x64-1.20.1.tgz && "
        "mv onnxruntime-linux-x64-1.20.1 /root/third_party/onnxruntime && "
        "echo '/root/third_party/onnxruntime/lib' > /etc/ld.so.conf.d/onnxruntime.conf && "
        "ldconfig && "
        "rm -f /tmp/onnxruntime-*.tgz"
    )
    .pip_install(
        "fastapi[standard]", "uvicorn[standard]",
        "python-chess", "onnxruntime-gpu",
        "python-dotenv", "pydantic",
    )
    .add_local_dir("engine", remote_path="/root/engine", copy=True,
                   ignore=["build/**", "**/*.o", "**/*.a"])
    # No -DONNXRUNTIME_ROOT flag: CMakeLists.txt hardcodes the relative path,
    # and we've placed the libs exactly where it expects them.
    .run_commands(
        "cd /root/engine && mkdir -p build && cd build && "
        "cmake .. -DCMAKE_BUILD_TYPE=Release -DPORTABLE_BUILD=ON && "
        "make -j$(nproc)"
    )
    .add_local_file("models/champion_fp32.onnx",
                    remote_path="/root/models/champion.onnx", copy=True)
    .add_local_dir("backend", remote_path="/root/backend", copy=True)
)

# ============================================================
# 3. Container class with snapshot initialization
# ============================================================
@app.cls(
    image=image,
    cpu=16.0,                       
    memory=8192,                   # 8 GB RAM
    enable_memory_snapshot=True,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    timeout=3600,
    volumes={"/data": games_volume},
    secrets=[chess_secrets],
)

@modal.concurrent(max_inputs=10)
class ChessBot:

    @modal.enter(snap=True)
    def setup(self):
        import os
        import sys

        sys.path.insert(0, "/root")

        os.environ["CHESS_ENGINE_BIN"] = "/root/engine/build/engine"
        os.environ["CHESS_MODEL_PATH"]  = "/root/models/champion.onnx"
        os.environ["CHESS_DATA_DIR"]    = "/data"
        os.environ["CHESS_DEVICE"]      = "cpu"

        from backend import app as backend_module

        if hasattr(backend_module, "engine") and backend_module.engine is None:
            if hasattr(backend_module, "_startup"):
                backend_module._startup()

        self.fastapi_app = backend_module.app
        print("[modal] snapshot init complete; engine is warm", flush=True)

    @modal.asgi_app()
    def web(self):
        return self.fastapi_app