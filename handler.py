"""RunPod Serverless entrypoint for llama.cpp.

Starts llama-server, waits for it to become healthy, then hands jobs to the
proxy. Runs at import time, matching the conventional RunPod worker shape --
the request-translation logic lives in proxy.py so it stays importable and
testable without launching anything.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

import runpod

import proxy
import server


def watchdog(process) -> None:
    """Exit the worker if llama-server dies, so RunPod recycles it.

    Without this the handler would keep accepting jobs and failing every one.
    """
    while True:
        code = process.poll()
        if code is not None:
            print(f"worker: llama-server exited with code {code}, stopping", flush=True)
            os._exit(1)
        time.sleep(5)


try:
    llama = server.start()
except Exception as exc:
    print(f"worker: startup failed: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1)

atexit.register(llama.terminate)
threading.Thread(target=watchdog, args=(llama,), daemon=True).start()

MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "1"))

runpod.serverless.start({
    "handler": proxy.handler,
    "return_aggregate_stream": True,
    "concurrency_modifier": lambda _current: MAX_CONCURRENCY,
})
