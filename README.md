# llama.cpp — RunPod Serverless worker

OpenAI-compatible GGUF inference on [llama.cpp](https://github.com/ggml-org/llama.cpp),
built for RunPod Serverless. Loads models from RunPod's model cache so there is
no download at cold start, supports vision projectors, and pins a known
llama.cpp build so architecture support is an explicit choice rather than
whatever existed on build day.

## Quick start

Deploy from the Hub, or point a Docker endpoint at an image built by this repo's
workflow. Then set the endpoint's **Model** field and three variables:

```bash
MODEL_REPO=Abiray/Qwen3.8-27B-Q4_K_M-GGUF     # same as the Model field
MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf
LLAMA_ARGS=--ctx-size 32768 --parallel 1 -ngl all --no-webui
```

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}}'
```

## Model sources

Set exactly one. The worker refuses to start if you set more than one.

| Source | Variables | When |
| --- | --- | --- |
| **RunPod model cache** | `MODEL_REPO` + `MODEL_FILE`, optional `MMPROJ_FILE` | Default. No download, not billed for the transfer, survives worker restarts |
| **Disk** | `MODEL_PATH`, optional `MMPROJ_PATH` | Network volume, or a model baked into a derived image |
| **Download at startup** | `HF_MODEL` (e.g. `unsloth/gemma-3-270m-it-GGUF:Q6_K`) | Trials and small models. Downloads on every cold start, on your clock |

For the cache, set the endpoint's **Model** field to the same value as
`MODEL_REPO` — that is what tells RunPod to mirror the repo to host disk before
the worker starts. Caching mirrors the *whole* repo, so prefer single-quant
repos; a repo holding twenty quants downloads all twenty.

## Settings

| Variable | Default | |
| --- | --- | --- |
| `LLAMA_ARGS` | `--ctx-size 4096 --parallel 1 -ngl all --no-webui` | Passed to `llama-server` |
| `STARTUP_TIMEOUT` | `600` | Seconds to wait for llama-server to report healthy |
| `MAX_CONCURRENCY` | `1` | Raise only alongside `--parallel` |
| `MODEL_NAME` | reported by llama-server | Model id echoed in responses |
| `HF_CACHE_DIR` | `/runpod-volume/huggingface-cache/hub` | Where RunPod puts cached snapshots |
| `REQUEST_TIMEOUT` | `3600` | Seconds per upstream request |

`-m`, `--mmproj`, `-hf`, `--host` and `--port` are built by the worker and are
rejected in `LLAMA_ARGS` rather than producing a duplicate argument.

Worth knowing before you copy flags from elsewhere: `--batch-size 2048`,
`--ubatch-size 512`, `--jinja` and `--no-context-shift` are all already upstream
defaults, and `--threads` is better left on auto.

## Requests

`messages` routes to `/v1/chat/completions`, `prompt` to `/v1/completions`, and
`openai_route` + `openai_input` reaches any other route directly:

```json
{"input": {"openai_route": "/v1/embeddings", "openai_input": {"input": "text"}}}
```

Vision works when `MMPROJ_FILE` is set, via the standard content-parts form:

```json
{"input": {"messages": [{"role": "user", "content": [
  {"type": "text", "text": "Describe this image."},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
]}], "max_tokens": 256}}
```

**Response shape.** This is a streaming handler, so RunPod returns `output` as a
list. A non-streaming request yields exactly one element:

```bash
... | jq '.output[0].choices[0].message.content'
```

Add `"stream": true` and read `/stream/<JOB_ID>` for chunks as they are
produced; `/runsync` still works and returns the whole list at the end.

## Choosing a llama.cpp build

`LLAMA_TAG` selects the base image, defaulting to `server-cuda-b10588`. This is
the one setting that matters over time: **llama.cpp must be newer than the model
architecture you want to run.** A build that predates a model fails at load with
an unknown-architecture error, and no amount of configuration fixes it.

Pick a tag from [ghcr.io/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp),
then update it in the place that matches how you deploy:

| Deploying via | To change llama.cpp |
| --- | --- |
| **Hub** | Edit the `ARG LLAMA_TAG` default in the `Dockerfile` and cut a release. RunPod builds from the repo and passes no build args, so the default is what ships |
| **Docker endpoint** (GHCR image) | **Actions → build worker image → Run workflow** with a new `llama_tag` |
| **Local** | `docker build --build-arg LLAMA_TAG=server-cuda-bXXXXX .` |

The workflow input only affects the GHCR image. It does not change what a Hub
listing serves — that comes from the `Dockerfile` default at release time.

The same rule covers GPUs: Blackwell (RTX 5090, `sm_120`) needs CUDA ≥ 12.8,
which the current base satisfies. `no kernel image is available for execution on
the device` means the build is too old for the card.

## Sizing the context

llama.cpp divides `--ctx-size` across `--parallel` slots, and the KV cache is
what usually decides whether a model fits. For Qwen3.8-27B Q4_K_M on a 32 GB
card — 16.81 GB of weights, 0.93 GB projector — the KV cache costs ~34 KiB per
token at `q8_0`:

| Context | KV cache | Total |
| --- | --- | --- |
| 32,768 | 1.14 GB | ~20 GB |
| 131,072 | 4.6 GB | ~24 GB |
| 262,144 | 9.13 GB | ~28.5 GB |

Above ~64k, add `-fa on -ctk q8_0 -ctv q8_0`. Flash attention is not optional
there: llama.cpp will not run a quantized V cache without it, and an f16 cache at
262,144 needs 17.18 GB and does not fit.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

```
handler.py          entrypoint: starts llama-server, registers the handler
proxy.py            job -> OpenAI route translation, streaming
server.py           model resolution, argv construction, readiness
Dockerfile          pinned llama.cpp base + this worker
.runpod/            Hub listing metadata and build-time tests
tests/              routing and argv checks, no GPU or model needed
```

Pushing to any branch runs the tests and verifies the image builds. It is
**published** only on a `v*` tag or a manual run — an ordinary push does not
overwrite anything anyone is running.

### Two images, two paths

These are independent, and it is worth knowing which one you are running:

- **RunPod's image.** Publishing to the Hub — or deploying via **New Endpoint →
  GitHub** — makes RunPod clone the repo, build the `Dockerfile` itself, and host
  the result in `registry.runpod.net`. The Hub indexes GitHub *releases*, not
  commits, so cut a release when you want the listing to update.
- **The GHCR image.** The workflow here builds
  `ghcr.io/<owner>/<repo>/llamacpp-worker`, for deploying via **New Endpoint →
  Docker**. It is pushed on a `v*` tag (as `1.2.3`, `latest` and `sha-…`) or by a
  manual run. The Hub never touches it.

Even if you only ever publish through the Hub, the workflow earns its place as a
build gate: it catches a broken `Dockerfile` in about four minutes, where a Hub
build failure is slower and harder to read. What it cannot do is test the worker
— GitHub runners have no GPU, so a green build proves the image assembles, not
that it serves. That is what `.runpod/tests.json` is for; RunPod runs it on a
real GPU at release time.

## Deploying from GitHub

**New Endpoint → GitHub** scans the repository's default branch for
`runpod.serverless.start()`. The worker must be merged to `main` — on a feature
branch the console reports `Could not find runpod.serverless.start() in your
repo`, and the same applies to Hub submissions, which build from releases cut on
the default branch.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `unknown model architecture` | llama.cpp older than the model — rebuild with a newer `LLAMA_TAG` |
| `no kernel image is available for execution on the device` | build too old for the GPU |
| `No cached snapshot for ...` | endpoint **Model** field empty, or it disagrees with `MODEL_REPO` |
| `MODEL_FILE=... is not in ...` | filename typo; the error lists what is actually there |
| `llama-server was not ready within Ns` | raise `STARTUP_TIMEOUT` |
| `LLAMA_ARGS must not set [...]` | a flag the worker builds itself |
| `Set only one model source` | more than one of `MODEL_REPO` / `MODEL_PATH` / `HF_MODEL` |
| `llama-server exited with code N, stopping` | it died after startup; the worker exits so RunPod recycles it |
| `Could not find runpod.serverless.start()` | the worker is not on the default branch — see above |
| OOM at high context | missing `-fa on`, so the V cache fell back to f16 |
