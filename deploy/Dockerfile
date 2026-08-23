# Serverless llama.cpp worker for Qwen3.8-27B (arch qwen3_5).
#
# The only change that matters versus upstream is the pinned base tag. The Hub's
# prebuilt worker images were built in May/June 2026, before Qwen3.8-27B existed,
# so their llama.cpp has no qwen3_5 support. Pin a build at or after b10430 --
# the build Abiray used to quantize the GGUF. b10588 is the newest published.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b10588

ENV PYTHONUNBUFFERED=1
WORKDIR /

RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common gpg-agent build-essential apt-utils git \
    && apt-get install --reinstall ca-certificates \
    && add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils bash curl \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# The RunPod handler + launcher. Replace this with your own fork and a pinned
# commit once you have one -- cloning a moving branch at build time is not
# reproducible.
ARG WORKER_REPO=https://github.com/ViniciosLugli/runpod-serverless
ARG WORKER_REF=main
RUN git clone --depth 1 --branch "${WORKER_REF}" "${WORKER_REPO}" /tmp/worker \
    && cp -r /tmp/worker/src /work \
    && rm -rf /tmp/worker

WORKDIR /work
RUN pip install -r ./requirements.txt && chmod +x /work/start.sh

ENTRYPOINT ["/work/start.sh"]
