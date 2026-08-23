# Deploy Qwen3.8-27B Q4_K_M on RunPod Serverless

Uses the worker in this repo, RunPod's model cache, and one RTX 5090.
For the reasoning behind the numbers see `llamacpp-runpod-qwen38-27b.md`;
for why the Hub's prebuilt workers don't work see `review.md`.

| | |
| --- | --- |
| Model | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` — 17.74 GB, two files, vision projector included |
| GPU | RTX 5090 (32 GB, pool `ADA_32_PRO`, $1.58/GPU-hr serverless, CUDA 12.8) |
| Context | 32,768 to start (~20 GB total). 262,144 is possible — see step 5 |
| Worker | this repo — `Dockerfile` + `src/`, built by GitHub Actions |

## Step 0 — why this repo exists

You cannot deploy this model from the RunPod Hub. Both published llama.cpp
workers build `FROM ghcr.io/ggml-org/llama.cpp:server-cuda` — a mutable tag
resolved at image build time — and their listed release images were built
2026-05-29 and 2026-06-14, before Qwen3.8-27B was released on 2026-08-05. The
llama.cpp inside them has no `qwen3_5` support and fails at model load.

So the image has to be built against a current llama.cpp. This repo does that
and nothing else: a pinned base, ~200 lines of worker, and a workflow.

## Step 1 — build the image

Push to any branch, or run **Actions → build worker image → Run workflow**. The
workflow runs the unit tests, then builds and pushes to:

```
ghcr.io/aaronbolton/runpod-testing/llamacpp-worker:latest
```

To rebuild against a newer llama.cpp, run the workflow manually and set
`llama_tag` (e.g. `server-cuda-b10600`). Anything ≥ `b10430` — the build the
GGUF was quantized with — should load this model. That input is the only knob
that matters over time; the rest of the image is stable.

**GHCR packages are private by default.** Either make the package public
(package settings → change visibility) or add a registry credential to the
RunPod endpoint, otherwise the worker cannot pull its own image.

## Step 2 — create the endpoint

**New Endpoint → Docker**, with the image from step 1.

| Setting | Value |
| --- | --- |
| GPU | RTX 5090 |
| Workers | min 0, max 1 |
| Container disk | 32 GB |
| Idle timeout | 120 s |
| Execution timeout | 600 s |
| **Model** | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` |

The **Model** field is the model cache. RunPod mirrors the whole repo snapshot
to host disk before the worker starts, so both the GGUF and the projector are
local and you are not billed for the download. No network volume needed.

## Step 3 — environment variables

Five. That is the whole configuration.

```bash
MODEL_REPO=Abiray/Qwen3.8-27B-Q4_K_M-GGUF
MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf
MMPROJ_FILE=mmproj-F16.gguf
LLAMA_ARGS=--ctx-size 32768 --parallel 1 -ngl all --no-webui
STARTUP_TIMEOUT=600
```

`MODEL_REPO` must match the endpoint's **Model** field — that is how the worker
finds the cached snapshot. `STARTUP_TIMEOUT` matters because 17 GB takes a while
to load and the worker will not wait forever by default.

Optional: `MAX_CONCURRENCY` (default 1), `MODEL_PATH` / `MMPROJ_PATH` to point at
files on a network volume instead of the cache, `HF_CACHE_DIR` if RunPod ever
moves the cache, `MODEL_NAME` to override the model id echoed in responses.

The worker builds `-m`, `--mmproj`, `--host` and `--port` itself and rejects
them in `LLAMA_ARGS` rather than silently producing a duplicate argument.

Note what is *not* in `LLAMA_ARGS`: `--batch-size`, `--ubatch-size`, `--jinja`
and `--no-context-shift` are all already the upstream defaults, and `--threads`
is better left on auto.

## Step 4 — check it came up

In the first worker's logs:

1. `worker: exec /app/llama-server -m /runpod-volume/huggingface-cache/hub/...`
   — cache hit, with `--mmproj` on the same line. If the path is wrong the
   worker fails immediately and lists the files it did find.
2. `n_ctx = 32768` in llama.cpp's context dump.
3. `worker: llama-server ready on http://127.0.0.1:8080`.

Then:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}}'
```

Vision, once text works:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}}
      ]}], "max_tokens": 256}}'
```

### Response shape

The handler is a streaming handler, so RunPod returns `output` as a **list**.
A non-streaming request yields exactly one element:

```bash
... | jq '.output[0].choices[0].message.content'
```

Add `"stream": true` to the input and use `/run` plus
`/stream/<JOB_ID>` to get chunks as they are produced; `/runsync` still works
and returns the whole list at the end.

`{"input": {"openai_route": "/v1/embeddings", "openai_input": {...}}}` reaches
any other llama-server route directly.

## Step 5 — only then, raise the context

32k costs 1.14 GB of KV cache. The model's full 262,144 costs 9.13 GB and needs
a quantized cache to fit at all:

```bash
LLAMA_ARGS=--ctx-size 262144 --parallel 1 -ngl all -fa on -ctk q8_0 -ctv q8_0 --no-webui
```

That lands at roughly 28.5 GB of 32 GB. `-fa on` is not optional there —
llama.cpp will not run a quantized V cache without flash attention — and
`--parallel` must stay at 1, since the context is divided across slots.

Intermediate values are the sensible move: 65,536 costs 2.3 GB, 131,072 costs
4.6 GB, both with room to spare.

## Failure modes

| Symptom | Cause |
| --- | --- |
| `unknown model architecture` / `qwen3_5` | base image predates the model — rebuild with a newer `llama_tag` |
| `no kernel image is available for execution on the device` | image lacks Blackwell `sm_120` kernels; newer `llama_tag` |
| `No cached snapshot for ...` | endpoint **Model** field empty, or `MODEL_REPO` disagrees with it |
| `MODEL_FILE=... is not in ...` | filename typo; the error lists what is actually in the snapshot |
| `llama-server was not ready within 600s` | raise `STARTUP_TIMEOUT` |
| `LLAMA_ARGS must not set [...]` | you passed a flag the worker builds itself |
| `llama-server exited with code N, stopping` | it died after startup; the worker exits so RunPod recycles it |
| OOM at 262k | `-fa on` missing, so the V cache fell back to f16 |
