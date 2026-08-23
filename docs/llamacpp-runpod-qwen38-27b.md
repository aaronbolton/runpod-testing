# Serverless llama.cpp on RunPod — Qwen3.8-27B Q4_K_M (vision, 262k ctx, RTX 5090)

Reference notes for a single-slot, inference-only llama.cpp serverless endpoint
with a pre-cached HuggingFace GGUF.

> **Start with [`deploy.md`](deploy.md), not this file.** This document's
> analysis holds up, but its deployment path does not: §1.3 offers two ways to
> reach a prebuilt Hub image that was built before Qwen3.8-27B existed and
> cannot load it. `deploy.md` is the working path. [`review.md`](review.md)
> records what was checked and what was wrong.

| Item | Value |
| --- | --- |
| Model repo | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` |
| Weights file | `Qwen3.8-27B-Q4_K_M.gguf` (16.81 GB) |
| Projector file | `mmproj-F16.gguf` (927 MB) |
| Base model | `Qwen/Qwen3.8-27B` (27.8B, arch `qwen3_5`, native vision-language) |
| GPU | RTX 5090 — 32 GB, pool `ADA_32_PRO`, serverless $1.58/GPU-hr |
| Context | 262,144 (equals the model's `max_position_embeddings`) |
| KV cache | `q8_0` for both K and V |
| Projector | enabled via `--mmproj` |
| Concurrency | 1 (`--parallel 1`, `MAX_CONCURRENCY=1`) |

---

## 1. Read this first — conflicts and corrections

**1.1 — Unsloth does not belong in this stack.** Unsloth is a PyTorch
fine-tuning accelerator (LoRA/QLoRA patching of `transformers`). It has no
inference path into llama.cpp, cannot load or accelerate GGUF, and installing
it in the worker adds a multi-GB torch dependency that never executes. The only
legitimate appearance of the name here is as a *quant publisher*, in three
different defaults that are easy to confuse:

| Where | Default for `LLAMA_SERVER_CMD_ARGS` |
| --- | --- |
| `Jacob-ML/inference-worker` v1.2.3 | `-hf unsloth/gemma-3-270m-it-GGUF:Q6_K --ctx-size 4096 -ngl 999` |
| `ViniciosLugli/runpod-serverless` v0.1.7 | `--ctx-size 4096 -ngl 999` — no `-hf` |
| `launcher.py`, variable unset | `-hf unsloth/gemma-3-270m-it-GGUF:IQ2_XXS --ctx-size 512 -ngl 999` |

All three name a HuggingFace repo hosting GGUF files, not the Unsloth runtime.
§4 always sets the variable, so none of them fires. Inference-only
alternatives if you were considering Unsloth for throughput: llama.cpp's own
flash-attention kernels (`-fa on`, already in this config), quantized KV cache
(already in this config), or a different serving engine (vLLM/SGLang) if you
are willing to drop GGUF and mmproj.

**1.2 — There is no first-party RunPod llama.cpp worker.** The Hub lists two
community llama.cpp serverless repos and neither is published by
`runpod-workers`:

| Hub repo | Deploys | mmproj support | Verdict |
| --- | --- | --- | --- |
| `Jacob-ML/inference-worker` | 1,795 | none — no mmproj env vars | Most popular, but you must smuggle `--mmproj-url` through the raw args |
| `ViniciosLugli/runpod-serverless` | 66 | first-class (`LLAMA_CACHED_MMPROJ_PATH`, `LLAMA_MMPROJ_PATH`, `LLAMA_MMPROJ_URL`) | **Use this one** — it is the mmproj-aware fork of the above |

This document targets `ViniciosLugli/runpod-serverless` because the projector
is a hard requirement. Both are built on `ghcr.io/ggml-org/llama.cpp:server-cuda`.

**1.3 — The listed Hub release will not offer you an RTX 5090.** *(Superseded:
both workarounds below run a prebuilt image that cannot load this model. See
`review.md` B1 and use `deploy.md`. The GPU-filtering analysis itself is
accurate.)* The currently
listed release (`v0.1.7`) pins `gpuIds: AMPERE_16,AMPERE_24,ADA_24`. The 5090
lives in pool `ADA_32_PRO` and is therefore filtered out of the Hub deploy
form, even though the repo's `main` branch `hub.json` has since added
`NVIDIA GeForce RTX 5090` to that list. Two workarounds, either is fine:

- Deploy from the Hub on any offered GPU, then open the endpoint and change
  the GPU selection to RTX 5090 (endpoint GPU config is editable after
  creation), **or**
- Skip the Hub form: create a custom template pointing at the prebuilt image
  `registry.runpod.net/vinicioslugli-runpod-serverless-main-dockerfile:cb40492c2`
  and pick the GPU freely. Every env var in §4 applies unchanged.

**1.4 — CUDA 12.8 is required, and it is what the template allows.** RTX 5090
is Blackwell `sm_120`, which needs CUDA ≥ 12.8. The template's
`allowedCudaVersions: ["12.8"]` is consistent with this. After first boot,
confirm the log has no `no kernel image is available for execution on the
device` — that string means the base image lacks `sm_120` kernels and you need
a newer `llama.cpp:server-cuda` tag.

**1.5 — Nothing to disable for training.** `llama-server` is an inference
binary; llama.cpp's training tooling (`llama-finetune`, `llama-export-lora`) is
a separate set of binaries not present in the server image. "Inference-only" is
therefore enforced by omission, not by a flag. See §6 for the explicit
verification list.

---

## 2. VRAM budget (32 GB card)

Qwen3.8-27B is a **hybrid** model: of its 64 layers, `layer_types` marks only
every 4th as `full_attention` (16 layers). Only those 16 carry a
context-proportional KV cache; the other 48 use a fixed-size recurrent state.
This is the reason a 262k context is viable on a single 32 GB card at all.

KV cache per token, full-attention layers only:

```
num_key_value_heads (4) × head_dim (256) × 2 (K and V) = 2,048 values/layer
2,048 × 16 full-attention layers                        = 32,768 values/token
q8_0 ≈ 1.0625 bytes/value (34-byte block per 32 values)
→ 34,816 bytes/token ≈ 34 KiB/token
× 262,144 tokens                                        ≈ 9.13 GB
```

| Component | VRAM |
| --- | --- |
| Weights (Q4_K_M) | 16.81 GB |
| Vision projector (F16) | 0.93 GB |
| KV cache @ 262,144, `q8_0` | 9.13 GB |
| Linear-attention recurrent state (48 layers, 1 seq) | ~0.15 GB |
| CUDA context + compute buffers @ `-ub 512` | ~1.2–1.8 GB |
| **Total** | **~28.2–28.8 GB of 32 GB** |

Roughly 3 GB of headroom. Two consequences:

- **`q8_0` is not an optimization here, it is a requirement.** At `f16` the
  same cache is 17.18 GB and the configuration does not fit.
- **Concurrency 1 is not a preference, it is the budget.** `--parallel 2`
  halves the per-slot context or doubles the cache; neither fits.

Keep `--ubatch-size` at 512. Raising it inflates the compute buffer directly
into the remaining headroom, and vision preprocessing takes transient VRAM on
top of the table above.

---

## 3. RunPod template / endpoint settings

| Setting | Value | Note |
| --- | --- | --- |
| Source | Hub → `ViniciosLugli/runpod-serverless` | or custom template, see §1.3 |
| Worker type | Serverless, GPU | |
| GPU | RTX 5090 (32 GB) | see §1.3 if not listed |
| GPU count | 1 | |
| CUDA version | 12.8 | |
| Container disk | 32 GB | sufficient — the cached snapshot lives on the host volume, not container disk. Raise to 60 GB **only** if you disable cached-model mode |
| **Model** (endpoint field) | `Abiray/Qwen3.8-27B-Q4_K_M-GGUF` | this is what triggers RunPod pre-caching; you are not billed for the download |
| Network volume | not required | |
| Max workers | 1–3 | your call |
| Active workers | 1 if you need low first-token latency; 0 otherwise | active workers bill continuously |
| Idle timeout | 60–300 s | a 17.7 GB reload is expensive; longer idle is usually cheaper than re-warming |
| Execution timeout | ≥ 600 s | 262k-token prefills are slow |
| Concurrency | 1 request/worker | enforced by `MAX_CONCURRENCY` below |

The whole repo is 17.7 GB across exactly two LFS files, so it caches cleanly —
this is the case the worker's own docs recommend (single-quant repos), as
opposed to multi-quant repos that are too large to cache reliably.

---

## 4. Environment variables

```bash
# --- model, served from the RunPod host cache ---
LLAMA_CACHED_MODEL=Abiray/Qwen3.8-27B-Q4_K_M-GGUF
LLAMA_CACHED_GGUF_PATH=Qwen3.8-27B-Q4_K_M.gguf
LLAMA_CACHED_MMPROJ_PATH=mmproj-F16.gguf

# --- llama-server arguments (see §5) ---
LLAMA_SERVER_CMD_ARGS=--ctx-size 262144 --parallel 1 -ngl 999 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 --ubatch-size 512 --no-context-shift --no-webui --jinja --threads 8

# --- worker behaviour ---
MAX_CONCURRENCY=1
RUNPOD_HANDLER_MODE=one-shot          # set to "stream" for token streaming
LLAMA_STARTUP_TIMEOUT_SECONDS=600     # default 120 is too short for a 17 GB load

# --- leave at defaults ---
LLAMA_CACHE_DIR=/runpod-volume/huggingface-cache/hub
LLAMA_SERVER_HOST=0.0.0.0
LLAMA_OPENAI_BASE_URL=http://localhost:3098/v1/
LLAMA_OPENAI_API_KEY=unused         # client-side only — NOT llama-server auth, see §9

# --- do NOT set ---
# LLAMA_MMPROJ_PATH / LLAMA_MMPROJ_URL — mutually exclusive with
#   LLAMA_CACHED_MMPROJ_PATH; setting two mmproj sources aborts startup.
```

### Launcher rules you must not violate

The worker builds argv as `[-m <resolved>] [--mmproj <resolved>] <your args> --port 3098`
and hard-fails on these:

- `--port` anywhere in `LLAMA_SERVER_CMD_ARGS` → error (3098 is worker-managed).
- `-m` / `--model` / `-hf` / `--hf-repo` / `-hfr` in the args while
  `LLAMA_CACHED_MODEL` is set → error.
- `--mmproj` / `--mmproj-url` in the args while any mmproj env var is set → error.
- More than one of `LLAMA_CACHED_MMPROJ_PATH`, `LLAMA_MMPROJ_PATH`,
  `LLAMA_MMPROJ_URL` → error.
- `LLAMA_CACHED_MODEL` set without `LLAMA_CACHED_GGUF_PATH` → error.

The projector is passed as `--mmproj` by the launcher from
`LLAMA_CACHED_MMPROJ_PATH`. That is the multimodal enablement — do not also
write it into the args.

---

## 5. Resolved llama-server command line

What actually runs inside the container:

```bash
/app/llama-server \
  -m /runpod-volume/huggingface-cache/hub/models--Abiray--Qwen3.8-27B-Q4_K_M-GGUF/snapshots/<rev>/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /runpod-volume/huggingface-cache/hub/models--Abiray--Qwen3.8-27B-Q4_K_M-GGUF/snapshots/<rev>/mmproj-F16.gguf \
  --ctx-size 262144 \
  --parallel 1 \
  -ngl 999 \
  -fa on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --no-context-shift \
  --no-webui \
  --jinja \
  --threads 8 \
  --host 0.0.0.0 \
  --port 3098
```

| Flag | Why |
| --- | --- |
| `--ctx-size 262144` | the requested context; matches the model's `max_position_embeddings` exactly, so no RoPE scaling is needed |
| `--parallel 1` | one KV slot gets the full 262k. With `--parallel N` the context is divided by N |
| `-ngl 999` | full offload; anything left on CPU destroys throughput at this context length |
| `-fa on` | **required** — llama.cpp will not run a quantized V cache without flash attention. Also the largest single memory saving in the attention path |
| `--cache-type-k q8_0` / `--cache-type-v q8_0` | the requested q8 precision; halves 17.18 GB to 9.13 GB and is what makes this fit (§2) |
| `--batch-size 2048` / `--ubatch-size 512` | *both are the upstream defaults — these restate them.* Leaving them off is identical; raising `-ub` inflates the compute buffer into the 3 GB of headroom |
| `--no-context-shift` | *no-op — context shift is already disabled by default.* Harmless, but it is not what keeps an overflow clean |
| `--no-webui` | no interactive UI on a serverless endpoint; the worker talks to `/v1/*` over localhost |
| `--jinja` | *no-op — enabled by default.* Does select the GGUF's embedded chat template, which is what Qwen3.8 turn formatting and tool calls need |
| `--threads 8` | *pins what the default (`-1`, auto) would pick better.* Irrelevant with full offload — drop it |

---

## 6. Inference-only verification

`llama-server` exposes no training or fitting mode, so this is a checklist of
things that must be **absent**, not flags to add:

- No `unsloth`, `peft`, `trl`, `bitsandbytes`, or `torch` in the image. The
  base is `ghcr.io/ggml-org/llama.cpp:server-cuda` plus `runpod` and `openai`
  Python packages — confirm you have not extended it.
- No `--lora` / `--lora-scaled` / `--lora-init-without-apply`. Adapter loading
  is not training, but it is state you did not ask for; leave it off.
- No `--slot-save-path`. Without it, slot state save/restore endpoints are
  disabled and the worker is fully stateless between jobs.
- `--no-webui` set, so no interactive surface is exposed.
- No `-hf` in the args — with cached mode on, this both errors out and would
  reintroduce a runtime download.
- `LLAMA_SERVER_CMD_ARGS` actually reached the worker. If the log shows a 270M
  gemma loading, it did not, and `launcher.py` fell back to its
  `-hf unsloth/gemma-3-270m-it-GGUF:IQ2_XXS` default.

---

## 7. Invoking the endpoint

The handler accepts OpenAI-shaped payloads inside RunPod's `input` wrapper and
proxies them to the local `llama-server`. If `messages` is present it routes to
`/v1/chat/completions`; if `prompt` is present, to `/v1/completions`. `model`
is filled in automatically from the loaded model when omitted.

Text:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Summarize the attached contract."}],
      "max_tokens": 1024,
      "temperature": 0.7
    }
  }'
```

Vision — this is the path that exercises `--mmproj`:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}}
        ]
      }],
      "max_tokens": 512
    }
  }'
```

Explicit route passthrough, for anything the shorthand does not cover:

```json
{"input": {"openai_route": "/v1/chat/completions", "openai_input": {"messages": [], "max_tokens": 256}}}
```

Set `RUNPOD_HANDLER_MODE=stream` and `"stream": true` for incremental tokens;
in `one-shot` mode the response is returned whole.

---

## 8. Post-deploy checks

Read the first worker's logs and confirm, in order:

1. `launcher.py: Running /app/llama-server -m /runpod-volume/huggingface-cache/hub/...`
   — cached mode resolved. If you instead see a download, the endpoint **Model**
   field or `LLAMA_CACHED_MODEL` is wrong and you are paying for the transfer.
2. `--mmproj` present on that same line, and a `clip`/vision encoder load
   further down — the projector is live.
3. `n_ctx = 262144` and `type_k = q8_0`, `type_v = q8_0` in the context dump.
4. KV cache size reported around **8.5 GiB / 9.13 GB**. A number near 17 GB
   means the `--cache-type-*` flags did not apply and you will OOM.
5. No `no kernel image is available for execution on the device` (§1.4).
6. `main: server is listening on http://0.0.0.0:3098` before the 600 s timeout.

Failure modes worth naming:

| Symptom | Cause |
| --- | --- |
| OOM during load | `-fa` off (V cache falls back to f16), or `--ubatch-size` raised |
| OOM on first vision request | headroom consumed by a high-resolution image; cap input image dimensions |
| Worker exits code 1 at startup | launcher validation — re-read the rules in §4 |
| `llama-server did not start within N seconds` | `LLAMA_STARTUP_TIMEOUT_SECONDS` still at its 120 s default |
| Requests queue instead of running in parallel | expected; `MAX_CONCURRENCY=1` is deliberate |

---

## 9. Web UI exposure and authentication

Short answer: **the web UI is off and unreachable, and no llama-server API key
is set.** The endpoint's only authentication is RunPod's.

### 9.1 — Web UI

Upstream default is `--ui, --webui, --no-ui, --no-webui` → *enabled*
(env `LLAMA_ARG_UI`). §4 passes `--no-webui`, so the UI is not built into the
running server at all.

That flag is belt-and-braces, not the actual barrier. A queue-based serverless
endpoint routes no inbound traffic to the container: requests arrive at
`api.runpod.ai/v2/<ENDPOINT_ID>/run*`, RunPod hands them to `handler.py`, and
the handler proxies to `http://localhost:3098/v1/`. There is no public route to
`/` — nothing outside the container can reach port 3098 regardless of the flag.

Two things would change that, and neither is in this configuration:

- The endpoint setting **Expose HTTP/TCP ports** (off by default) publishes the
  worker's IP and port for direct external traffic. Leave it off. This worker is
  a queue handler, not a load-balancing worker, so turning it on gains nothing.
- Running this image on a **Pod** instead of a serverless endpoint. Note that
  `LLAMA_SERVER_HOST=0.0.0.0` means llama-server listens on every container
  interface, not just loopback — inert while no port is published, but on a Pod
  with an exposed HTTP port that is a fully open, unauthenticated server. If you
  ever move this to a Pod, set an API key first (§9.3).

### 9.2 — `LLAMA_OPENAI_API_KEY` is not server authentication

This is the most common misreading of the worker's env list. The variable is
read in exactly one place — `engine.py`:

```python
base_url=os.getenv("LLAMA_OPENAI_BASE_URL", "http://localhost:3098/v1/"),
api_key=os.getenv("LLAMA_OPENAI_API_KEY", "unused"),
```

It is the *client* credential the worker's OpenAI SDK object is constructed
with, and the SDK refuses to instantiate with an empty one — hence the
placeholder. It becomes an `Authorization: Bearer unused` header on the
container-local hop. The hub.json description says as much: *"API key for the
local OpenAI client. llama-server does not require one by default."*

### 9.3 — llama-server has no key, and how to give it one

`--api-key KEY` (env `LLAMA_API_KEY`) is **default: none**, and neither §4 nor
the launcher sets it — `launcher.py` injects only `--host` and `--port 3098`.
So llama-server accepts every request on 3098 unauthenticated and ignores the
`Bearer unused` header entirely. With no port published, the only clients on
that interface are the handler and anything else in the same container, so this
is acceptable as configured.

If you want the local hop authenticated anyway, set **both** halves to the same
secret — llama-server picks its side up from the environment, so nothing goes
into `LLAMA_SERVER_CMD_ARGS`:

```bash
LLAMA_API_KEY=<secret>          # read by llama-server → same as --api-key
LLAMA_OPENAI_API_KEY=<secret>   # read by the handler's OpenAI client
```

Set only the first and every request 401s with
`{"error": {"code": 401, "message": "Invalid API Key", ...}}`; `GET /health` is
the one public route. Setting only the second changes nothing.

### 9.4 — The authentication that does apply

Callers authenticate to RunPod, not to llama.cpp:

```
Authorization: Bearer $RUNPOD_API_KEY   →   api.runpod.ai/v2/<ENDPOINT_ID>/runsync
```

That header is the perimeter. Scope the key to the endpoint and rotate it in
the RunPod console; anyone holding it can invoke the model, and no llama.cpp
setting constrains that.

### 9.5 — Related surfaces, all off

| Surface | State | Why |
| --- | --- | --- |
| Web UI | disabled | `--no-webui` (§4), and no public route regardless |
| CORS | irrelevant | no browser ever reaches 3098; llama.cpp's permissive default reflects any `Origin`, which only matters on a published port |
| `--tools` / `--agent` | not set | these expose filesystem read/write over the API — never enable them on a shared endpoint |
| `--ui-mcp-proxy` | not set | upstream default disabled; its own docs warn against untrusted environments |
| `--slot-save-path` | not set | slot save/restore endpoints stay disabled (§6) |
| `--path` | not set | no static file serving |
