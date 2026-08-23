# runpod-testing

A llama.cpp worker for RunPod Serverless, built to run **Qwen3.8-27B Q4_K_M**
(GGUF, vision) from RunPod's model cache.

```
Dockerfile                     pinned llama.cpp base + this worker
src/server.py                  resolve the cached model, run llama-server
src/handler.py                 RunPod job -> OpenAI route on localhost
src/main.py                    entrypoint
tests/                         routing and argv checks, no GPU needed
.github/workflows/build-image.yml   test, build, push to GHCR
```

- **[docs/deploy.md](docs/deploy.md)** — the deployment. Start here.
- [docs/review.md](docs/review.md) — review findings, verified against primary
  sources.
- [docs/llamacpp-runpod-qwen38-27b.md](docs/llamacpp-runpod-qwen38-27b.md) —
  reference notes: VRAM budget, exposure and auth.

The reason this worker exists rather than a Hub deploy: the prebuilt llama.cpp
workers on the RunPod Hub were built before this model was released and cannot
load it. See `docs/review.md`.

## Local checks

```bash
python -m unittest discover -s tests -v
```
