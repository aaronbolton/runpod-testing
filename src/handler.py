"""RunPod handler: proxies OpenAI-shaped jobs to the local llama-server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterator

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
