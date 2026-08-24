"""RunPod Serverless handler for llama.cpp.

Brings up llama-server, then proxies each job to its OpenAI-compatible routes.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

import runpod

import server

CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "3600"))


def _route(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pick the llama-server route and body for a job input."""
    if "openai_route" in payload:
        return payload["openai_route"], dict(payload.get("openai_input") or {})

    body = {k: v for k, v in payload.items() if k not in ("openai_route", "openai_input")}
    if "messages" in body:
        return CHAT, body
    if "prompt" in body:
        return COMPLETIONS, body
    raise ValueError("input needs one of 'messages', 'prompt', or 'openai_route'")


def _request(route: str, body: dict[str, Any]):
    req = urllib.request.Request(
        server.BASE_URL + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _error(exc: urllib.error.HTTPError) -> dict[str, Any]:
    detail = exc.read().decode(errors="replace")
    try:
        return json.loads(detail)
    except ValueError:
        return {"error": {"code": exc.code, "message": detail or exc.reason}}


def _stream(route: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    with _request(route, body) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                return
            try:
                yield json.loads(chunk)
            except ValueError:
                continue


def handler(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yields one full response, or many chunks when the job sets stream: true.

    Registered with return_aggregate_stream, so /runsync and /run collect the
    yields into a list while /stream delivers them as they arrive.
    """
    payload = job.get("input") or {}

    try:
        route, body = _route(payload)
    except ValueError as exc:
        yield {"error": {"message": str(exc)}}
        return

    body.setdefault("model", server.model_id())

    try:
        if body.get("stream"):
            yield from _stream(route, body)
        else:
            with _request(route, body) as resp:
                yield json.load(resp)
    except urllib.error.HTTPError as exc:
        yield _error(exc)
    except (urllib.error.URLError, OSError) as exc:
        yield {"error": {"message": f"llama-server unreachable: {exc}"}}


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
            "handler": handler,
            "return_aggregate_stream": True,
            "concurrency_modifier": lambda _current: concurrency,
        }
    )


if __name__ == "__main__":
    main()
