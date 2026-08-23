"""Resolve a GGUF model and run llama-server as a subprocess.

llama-server is bound to loopback: nothing outside the container talks to it,
only the handler in this same process.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

BIN = os.environ.get("LLAMA_BIN", "/app/llama-server")
HOST = "127.0.0.1"
PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE_URL = f"http://{HOST}:{PORT}"
DEFAULT_CACHE = "/runpod-volume/huggingface-cache/hub"

# Built by this module from the model settings. Passing them again in LLAMA_ARGS
# is a configuration error, not something to silently merge.
RESERVED = {
    "-m", "--model", "-hf", "--hf-repo", "-hfr",
    "--mmproj", "--mmproj-url", "--host", "--port",
}

_model_id: str | None = None


def _snapshot(repo: str, cache: Path) -> Path:
    """Locate the cached snapshot directory for a HuggingFace repo id."""
    org, sep, name = repo.partition("/")
    if not sep or not org or not name:
        raise ValueError(f"MODEL_REPO must look like 'org/name', got {repo!r}")

    root = cache / f"models--{org}--{name}"
    ref = root / "refs" / "main"
    if ref.is_file():
        candidate = root / "snapshots" / ref.read_text().strip()
        if candidate.is_dir():
            return candidate

    snapshots = root / "snapshots"
    dirs = sorted(p for p in snapshots.iterdir() if p.is_dir()) if snapshots.is_dir() else []
    if dirs:
        return dirs[0]

    raise FileNotFoundError(
        f"No cached snapshot for {repo!r} under {cache}. Set the endpoint's "
        f"Model field to {repo!r} so RunPod caches it before the worker starts."
    )


def _pick(snapshot: Path, filename: str, label: str) -> str:
    path = snapshot / filename
    if path.is_file():
        return str(path)
    present = sorted(p.name for p in snapshot.iterdir() if p.is_file())
    raise FileNotFoundError(f"{label}={filename!r} is not in {snapshot}. Present: {present}")


def _model_flags() -> list[str]:
    """One of three sources: the RunPod cache, a path on disk, or -hf download."""
    repo = os.environ.get("MODEL_REPO")
    path = os.environ.get("MODEL_PATH")
    hf = os.environ.get("HF_MODEL")

    chosen = [n for n, v in (("MODEL_REPO", repo), ("MODEL_PATH", path), ("HF_MODEL", hf)) if v]
    if len(chosen) > 1:
        raise ValueError(f"Set only one model source, got {chosen}.")
    if not chosen:
        raise ValueError(
            "Set MODEL_REPO + MODEL_FILE to use RunPod's model cache, MODEL_PATH for "
            "a file already on disk, or HF_MODEL to download at startup."
        )

    if hf:
        # llama.cpp downloads this itself and picks up a projector automatically.
        return ["-hf", hf]

    if path:
        flags = ["-m", path]
        mmproj = os.environ.get("MMPROJ_PATH")
        return flags + ["--mmproj", mmproj] if mmproj else flags

    model_file = os.environ.get("MODEL_FILE")
    if not model_file:
        raise ValueError("MODEL_REPO is set but MODEL_FILE is not — name the .gguf inside the repo.")

    snapshot = _snapshot(repo, Path(os.environ.get("HF_CACHE_DIR", DEFAULT_CACHE)))
    flags = ["-m", _pick(snapshot, model_file, "MODEL_FILE")]
    mmproj_file = os.environ.get("MMPROJ_FILE")
    if mmproj_file:
        flags += ["--mmproj", _pick(snapshot, mmproj_file, "MMPROJ_FILE")]
    return flags


def build_argv() -> list[str]:
    extra = shlex.split(os.environ.get("LLAMA_ARGS", ""))
    clashes = sorted({a for a in extra if a in RESERVED})
    if clashes:
        raise ValueError(
            f"LLAMA_ARGS must not set {clashes} — the worker manages those. "
            "Use MODEL_REPO / MODEL_FILE / MMPROJ_FILE / MODEL_PATH / HF_MODEL instead."
        )

    return [BIN, *_model_flags(), "--host", HOST, "--port", str(PORT), *extra]


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        # 503 while weights load, connection refused before the socket is up.
        return False


def start() -> subprocess.Popen:
    """Launch llama-server and block until it reports healthy."""
    argv = build_argv()
    print("worker: exec " + " ".join(shlex.quote(a) for a in argv), flush=True)

    proc = subprocess.Popen(argv, cwd="/app")
    timeout = float(os.environ.get("STARTUP_TIMEOUT", "600"))
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"llama-server exited with code {code} during startup")
        if _healthy():
            print(f"worker: llama-server ready on {BASE_URL}", flush=True)
            return proc
        time.sleep(2)

    proc.kill()
    raise TimeoutError(
        f"llama-server was not ready within {timeout:.0f}s. Large models need a "
        "higher STARTUP_TIMEOUT."
    )


def model_id() -> str:
    """The model name llama-server reports, used when a request omits one."""
    global _model_id
    if _model_id is None:
        _model_id = os.environ.get("MODEL_NAME") or ""
        if not _model_id:
            try:
                with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
                    _model_id = json.load(resp)["data"][0]["id"]
            except Exception:
                _model_id = "local"
    return _model_id
