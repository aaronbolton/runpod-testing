"""Entrypoint: bring up llama-server, then serve RunPod jobs against it."""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

import runpod

import handler
import server


def _watchdog(proc) -> None:
    """Exit the worker if llama-server dies, so RunPod recycles it.

    Without this the handler would keep accepting jobs and failing every one.
    """
    while True:
        code = proc.poll()
        if code is not None:
            print(f"worker: llama-server exited with code {code}, stopping", flush=True)
            os._exit(1)
        time.sleep(5)


def main() -> None:
    try:
        proc = server.start()
    except Exception as exc:
        print(f"worker: startup failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    atexit.register(proc.terminate)
    threading.Thread(target=_watchdog, args=(proc,), daemon=True).start()

    concurrency = int(os.environ.get("MAX_CONCURRENCY", "1"))
    runpod.serverless.start(
        {
            "handler": handler.handler,
            "return_aggregate_stream": True,
            "concurrency_modifier": lambda _current: concurrency,
        }
    )


if __name__ == "__main__":
    main()
