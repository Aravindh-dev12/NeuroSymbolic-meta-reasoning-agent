# Production Dockerfile for the NeuroSymbolic Meta-Reasoning Agent.

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LLM_BACKEND=local \
    LOCAL_MODEL_NAME=auto \
    VECTOR_DB_TYPE=sqlite

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY pyproject.toml README.md ./
COPY neurosymbolic_agent/ ./neurosymbolic_agent/
COPY app.py ./

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -e ".[dashboard,local-llm]"

RUN mkdir -p logs data models cache

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD neuro-agent models >/dev/null || exit 1

CMD ["agent", "serve"]

FROM ubuntu:22.04 AS cpu-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_BACKEND=local \
    LOCAL_MODEL_NAME=auto \
    VECTOR_DB_TYPE=sqlite

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY pyproject.toml README.md ./
COPY neurosymbolic_agent/ ./neurosymbolic_agent/
COPY app.py ./

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -e ".[dashboard,local-llm]"

RUN mkdir -p logs data models cache

EXPOSE 7860

CMD ["agent", "serve"]
