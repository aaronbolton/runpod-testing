# runpod-testing

Running **Qwen3.8-27B Q4_K_M** (GGUF, vision) on RunPod Serverless with llama.cpp
and RunPod's model cache.

- **[docs/deploy.md](docs/deploy.md)** — the deployment. Start here.
- [deploy/Dockerfile](deploy/Dockerfile) — worker image, pinned to a llama.cpp
  build that supports the `qwen3_5` architecture.
- [docs/review.md](docs/review.md) — review findings, verified against primary
  sources.
- [docs/llamacpp-runpod-qwen38-27b.md](docs/llamacpp-runpod-qwen38-27b.md) —
  reference notes: VRAM budget, launcher rules, exposure and auth.

The one thing worth knowing up front: the prebuilt llama.cpp workers on the
RunPod Hub were built before this model was released and cannot load it. You
have to rebuild the image against a current llama.cpp. It is a one-line change.
