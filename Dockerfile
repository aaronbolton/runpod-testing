# llama.cpp worker for RunPod Serverless.
#
# The base tag must postdate the model you intend to run: llama.cpp gains
# architecture support over time, and the Hub's prebuilt workers pull a mutable
# `server-cuda` tag that was resolved whenever their image happened to be built.
# b10430 is the build Qwen3.8-27B's GGUF was quantized with; b10588 is newer.
ARG LLAMA_TAG=server-cuda-b10588
FROM ghcr.io/ggml-org/llama.cpp:${LLAMA_TAG}

# Base is nvidia/cuda:12.8.1-runtime-ubuntu24.04 with /app/llama-server and its
# backend .so files in /app. It ships no Python, so add one in a venv rather
# than fighting PEP 668 on the system interpreter.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && python3 -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /worker/requirements.txt
RUN pip install -r /worker/requirements.txt

COPY src/ /worker/
WORKDIR /worker

ENTRYPOINT ["python", "-u", "main.py"]
