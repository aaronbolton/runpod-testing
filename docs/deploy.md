# Deploy Qwen3.8-27B Q4_K_M on RunPod Serverless

The short path. Uses RunPod's model cache, one RTX 5090, llama.cpp with the
vision projector. For the reasoning behind the numbers see
`llamacpp-runpod-qwen38-27b.md`; for what was wrong with the previous plan see
`review.md`.

| | |
| --- | --- |
| Model | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` — 17.74 GB, two files, vision projector included |
| GPU | RTX 5090 (32 GB, pool `ADA_32_PRO`, $1.58/GPU-hr serverless, CUDA 12.8) |
| Context | 32,768 to start (~20 GB total). 262,144 is possible — see step 5 |
| Worker | `ViniciosLugli/runpod-serverless`, rebuilt against current llama.cpp |

## Step 0 — why you have to build an image

You cannot deploy this from the Hub as-is. Both published llama.cpp workers were
built before Qwen3.8-27B was released (2026-05-29 and 2026-06-14 vs 2026-08-05),
and their Dockerfiles pull the mutable `ghcr.io/ggml-org/llama.cpp:server-cuda`
tag at build time. The llama.cpp inside them has no `qwen3_5` support and will
fail at model load.

The fix is one line — pin the base image to a build that postdates the model.
Everything else about the worker is fine.

## Step 1 — get an image built

**Option A (no local Docker).** Fork `ViniciosLugli/runpod-serverless`, change
line 2 of its `Dockerfile`:

```diff
-FROM ghcr.io/ggml-org/llama.cpp:server-cuda
+FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b10588
```

Then in RunPod: **New Endpoint → GitHub**, pick your fork. RunPod builds it.
(Limits: 30 min for `docker build`, 80 GB image. This image is well inside both.)

**Option B (local Docker).** `deploy/Dockerfile` in this repo does the same
thing standalone:

```bash
docker build -t <your-registry>/llamacpp-qwen38:b10588 deploy/
docker push  <your-registry>/llamacpp-qwen38:b10588
```

Then **New Endpoint → Docker** with that image.

Pin a newer `server-cuda-b*` tag if one has shipped; anything ≥ `b10430` (the
build the GGUF was quantized with) should load the model.

## Step 2 — endpoint settings

| Setting | Value |
| --- | --- |
| GPU | RTX 5090 |
| Workers | min 0, max 1 |
| Container disk | 32 GB |
| Idle timeout | 120 s |
| Execution timeout | 600 s |
| **Model** | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` |

The **Model** field is the model cache. RunPod mirrors the whole repo snapshot
to the host before the worker starts, so both the GGUF and the projector are on
local disk and you are not billed for the download. No network volume needed.

## Step 3 — environment variables

Six. That is the whole configuration.

```bash
LLAMA_CACHED_MODEL=Abiray/Qwen3.8-27B-Q4_K_M-GGUF
LLAMA_CACHED_GGUF_PATH=Qwen3.8-27B-Q4_K_M.gguf
LLAMA_CACHED_MMPROJ_PATH=mmproj-F16.gguf
LLAMA_SERVER_CMD_ARGS=--ctx-size 32768 --parallel 1 -ngl all --no-webui
LLAMA_STARTUP_TIMEOUT_SECONDS=600
MAX_CONCURRENCY=1
```

Everything else stays at its default. `LLAMA_STARTUP_TIMEOUT_SECONDS` is the one
non-obvious entry: the default of 120 s is not enough to load 17 GB and the
worker will kill itself mid-load without it.

Four rules the launcher enforces, all of which will abort startup:

- no `--port` in the args (3098 is worker-managed)
- no `-m` / `--model` / `-hf` in the args while `LLAMA_CACHED_MODEL` is set
- no `--mmproj` in the args while `LLAMA_CACHED_MMPROJ_PATH` is set
- only one of the three `*MMPROJ*` variables at a time

Note what is *not* in the args: `--batch-size`, `--ubatch-size`, `--jinja` and
`--no-context-shift` are all already the upstream defaults, and `--threads` is
better left on auto.

## Step 4 — check it came up

In the first worker's logs:

1. `Running /app/llama-server -m /runpod-volume/huggingface-cache/hub/...` —
   cache hit. If you see a download instead, the **Model** field is wrong and
   you are paying for the transfer.
2. `--mmproj` on that same line, and a vision encoder load below it.
3. `n_ctx = 32768`.
4. `server is listening on http://0.0.0.0:3098`.

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

## Step 5 — only then, raise the context

32k costs 1.14 GB of KV cache. The model's full 262,144 costs 9.13 GB and needs
a quantized cache to fit at all:

```bash
LLAMA_SERVER_CMD_ARGS=--ctx-size 262144 --parallel 1 -ngl all -fa on -ctk q8_0 -ctv q8_0 --no-webui
```

That lands at roughly 28.5 GB of 32 GB. `-fa on` is not optional there —
llama.cpp will not run a quantized V cache without flash attention — and
`--parallel` must stay at 1, since the context is divided across slots.

Intermediate values are the sensible move: 65,536 costs 2.3 GB, 131,072 costs
4.6 GB, both with room to spare.

## Failure modes

| Symptom | Cause |
| --- | --- |
| `unknown model architecture` / `qwen3_5` at load | base image predates the model — step 1 |
| `no kernel image is available for execution on the device` | image lacks Blackwell `sm_120` kernels; use a newer `server-cuda-b*` |
| `llama-server did not start within N seconds` | `LLAMA_STARTUP_TIMEOUT_SECONDS` still at 120 |
| Worker exits 1 immediately | launcher validation — recheck the four rules in step 3 |
| Model downloads on every cold start | **Model** field empty or misspelled |
| OOM at 262k | `-fa on` missing, so the V cache fell back to f16 |
