# Project review — findings

Review of `docs/llamacpp-runpod-qwen38-27b.md` against primary sources
(HuggingFace API, the RunPod Hub API, RunPod docs, the worker's source, and the
upstream `llama.cpp` server README) on 2026-08-23.

**Verdict: the analysis is sound but the deployment path does not work.** The
VRAM budget, the model choice and the launcher rules all check out against the
sources. The one thing that fails is the step where you actually deploy: every
prebuilt image the guide points at was built before the model existed.

---

## Blocking

### B1 — Both prebuilt llama.cpp worker images predate the model

This is the finding that invalidates §1.3 and §3 of the guide.

| Artifact | Built | Source |
| --- | --- | --- |
| `Qwen/Qwen3.8-27B` released | **2026-08-05** | HF repo `createdAt` |
| `Abiray/…-Q4_K_M-GGUF` quantized with llama.cpp `b10430` | 2026-08-14 | repo README + `lastModified` |
| `ViniciosLugli/runpod-serverless` v0.1.7 image `…:cb40492c2` | **2026-05-29** | Hub API `listedRelease.createdAt` |
| `Jacob-ML/inference-worker` v1.2.3 image `…:029f37afa` | **2026-06-14** | Hub API `listedRelease.createdAt` |

The worker Dockerfile is `FROM ghcr.io/ggml-org/llama.cpp:server-cuda` — an
unpinned, mutable tag resolved **at image build time**. The listed release was
built 10 weeks before Qwen3.8-27B was published, so the `llama.cpp` baked into
it cannot contain the `qwen3_5` architecture. Loading the GGUF will fail at
model-load with an unknown-architecture error, not an OOM or a config problem.

Neither of the guide's §1.3 workarounds helps: option 1 (deploy from the Hub,
then change GPU) and option 2 (custom template pointing at
`registry.runpod.net/vinicioslugli-runpod-serverless-main-dockerfile:cb40492c2`)
both run that same May image. **There is no zero-build path.** The image has to
be rebuilt against a current `llama.cpp`. See `docs/deploy.md`.

Fix: pin the base to a tag that postdates the model. `server-cuda-b10588` is
the newest published build (503 `server-cuda-b*` tags exist on GHCR; 9 are
≥ `b10430`, the build the GGUF was quantized with).

---

## Significant

### S2 — Most of `LLAMA_SERVER_CMD_ARGS` restates upstream defaults

Checked against the current server README's argument table:

| Flag in the guide | Upstream default | Verdict |
| --- | --- | --- |
| `--batch-size 2048` | `2048` | no-op |
| `--ubatch-size 512` | `512` | no-op |
| `--jinja` | **enabled** | no-op |
| `--no-context-shift` | context shift **already disabled** | no-op |
| `--threads 8` | `-1` (auto) | pins to a worse value than auto |
| `-ngl 999` | `auto`; `all` is the modern spelling | works, dated |
| `-fa on` | `auto` | worth keeping explicit |
| `-ctk/-ctv q8_0` | `f16` | **required** at long context |
| `--ctx-size` | `0` = from model (262144) | worth setting explicitly, downward |

The guide's §5 rationale table justifies several of these as if they were doing
work. `--no-context-shift` is the clearest case: it is presented as protection
against silent degradation on a hybrid architecture, but context shift is off by
default, so the flag changes nothing.

### S3 — The 262,144 context is the most expensive possible starting point

The arithmetic is right (verified below), but it lands at ~28.2–28.8 GB of 32 GB
and the guide then has to spend three sections defending that headroom. For a
first deployment, `--ctx-size 32768` costs 1.14 GB of KV instead of 9.13 GB,
totals ~20 GB, removes the OOM failure modes entirely, and needs no
`-ctk/-ctv` flags at all. Raise it once the endpoint is known-good.

### S4 — `Abiray` is a low-profile publisher and the guide never says so

39.8K downloads, 16 likes, no imatrix, no MTP file. It is nonetheless the
**correct** choice, for a reason the guide gets right but under-argues: RunPod
model caching "downloads all quantization versions" in a repo. The obvious
alternative, `unsloth/Qwen3.8-27B-GGUF` (6.7M downloads), holds 26 quants plus
BF16 — roughly 450 GB — and would be cached in full. Abiray's repo is exactly
two LFS files totalling 17.74 GB. Worth stating that the trade is deliberate.

---

## Minor

- **M5** — §1.1's claim about the stock default was right about the wrong repo,
  and my earlier "correction" to it was also wrong. Three distinct values:
  Jacob-ML v1.2.3 defaults to `-hf unsloth/gemma-3-270m-it-GGUF:Q6_K --ctx-size
  4096 -ngl 999`; ViniciosLugli v0.1.7 defaults to `--ctx-size 4096 -ngl 999`
  with no `-hf`; `launcher.py`'s fallback for an unset variable is `-hf
  unsloth/gemma-3-270m-it-GGUF:IQ2_XXS --ctx-size 512 -ngl 999`. Corrected in
  place.
- **M6** — §3 says container disk 32 GB is "sufficient". True, and it is also
  the only value the Hub release offers (`containerDiskInGb: 32`).
- **M7** — The model has `mtp_num_hidden_layers: 1`. Abiray ships no MTP file,
  so speculative decoding is unavailable on this quant. Not required, but it is
  a real capability the chosen repo gives up.
- **M8** — `LLAMA_MMPROJ_URL`'s own description warns that RunPod caching may
  only cache the main GGUF. It does not: caching mirrors the whole repo
  snapshot, so `LLAMA_CACHED_MMPROJ_PATH` is correct here.

---

## Verified correct

Worth recording, because these were the claims most likely to be wrong:

- **The KV cache arithmetic.** `config.json` confirms 64 layers with
  `full_attention_interval: 4` and a `layer_types` array holding exactly 16
  `full_attention` entries, `num_key_value_heads: 4`, `head_dim: 256`,
  `max_position_embeddings: 262144`. So 4 × 256 × 2 × 16 = 32,768 values/token;
  at q8_0's 34 bytes per 32 values that is 34 KiB/token, and 9.13 GB at 262,144
  tokens. The guide's central claim — that a 262k context fits on one 32 GB card
  only because 48 of 64 layers are linear-attention — is correct.
- **The recurrent state estimate.** `linear_num_value_heads: 48` ×
  `linear_key_head_dim: 128` × `linear_value_head_dim: 128` × fp32 × 48 layers
  ≈ 0.15 GB. Matches.
- **File sizes.** 16,810,714,400 + 927,607,552 = 17.74 GB across two LFS files.
- **RTX 5090 = pool `ADA_32_PRO`, 32 GB, $1.58/hr serverless, CUDA 12.8.**
  Exact match, and serverless availability is currently HIGH.
- **v0.1.7 pins `gpuIds: AMPERE_16,AMPERE_24,ADA_24`** while `main`'s `hub.json`
  has since added `NVIDIA GeForce RTX 5090`. §1.3's premise was right.
- **Every launcher validation rule in §4.** All five match `launcher.py`.
- **The Unsloth analysis (§1.1) and the inference-only checklist (§6).**
- **§9 on the web UI and API keys** (added in the previous commit).
