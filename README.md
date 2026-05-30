# NeuroSymbolic Meta-Reasoning Agent

A CLI-first agentic reasoning system that combines local/open-weight LLM routing, symbolic solvers, neural inference, episodic memory, hierarchical planning, recursive self-critique, and constitutional safety checks.

## What Is Included

- Installed CLI commands: `agent`, `neuro-agent`, and `neurosymbolic-agent`
- Local model routing through Ollama, llama.cpp GGUF models, or Hugging Face Transformers
- Offline structured fallback so the CLI can still run when no local LLM runtime is available
- Symbolic reasoning with Z3-style logic, SymPy math generation, arithmetic fallback, and a knowledge base
- Neural task routing with sentence embeddings and a lightweight task classifier
- Memory via working memory, episodic JSON storage, and Chroma/FAISS/SQLite vector stores
- Recursive planning, critique/self-improvement, telemetry, traces, and constitutional guardrails
- Gradio dashboard, Docker support, GitHub workflow, and Kaggle publishing helpers

## Quick Start

```bash
cd C:\Users\ELCOT\Downloads\NeuroSymbolic-meta-reasoning-agent
python -m pip install -e ".[dashboard,dev]"
agent run "All mammals breathe air. Whales are mammals. Do whales breathe air?"
```

Interactive mode:

```bash
agent start
```

The same CLI is also available as `neuro-agent` and `neurosymbolic-agent`.

List available local/open-weight model backends:

```bash
agent models
```

Check your runtime:

```bash
agent doctor
```

Launch the dashboard:

```bash
agent serve
```

## Local Open-Weight Models

The default config uses `LOCAL_MODEL_NAME=auto`. Auto mode prefers Ollama when it is already running, otherwise it tries the llama.cpp model catalog. If local model startup fails, the agent falls back to `agentic-rules`, a deterministic structured fallback that keeps the CLI usable.

Useful model choices:

```bash
agent --model ollama-qwen2.5 run "Create a plan to test this agent"
agent --model ollama-llama3.1 run "Explain neuro-symbolic reasoning"
agent --model ollama-deepseek-r1 run "Solve a multi-step logic puzzle"
agent --model transformers-qwen2.5-0.5b run "Summarize this design"
agent --model llama3-8b run "Reason through a formal proof"
```

For Ollama:

```bash
ollama serve
ollama pull qwen2.5:7b
agent --model ollama-qwen2.5 run "Solve 2x + 3 = 7"
```

For llama.cpp GGUF:

```bash
python -m pip install llama-cpp-python
agent --model llama3-8b run "Solve 15 * 7 + 23"
```

## Configuration

Config lives in `neurosymbolic_agent/configs/agent_config.yaml`. Environment variables override YAML:

```bash
set LLM_BACKEND=local
set LOCAL_MODEL_NAME=auto
set VECTOR_DB_TYPE=sqlite
set ENABLE_TELEMETRY=true
```

Cloud backends are still supported:

```bash
set ANTHROPIC_API_KEY=your_key
agent --backend anthropic run "Classify this sentiment: the product is fast"
```

## Docker

```bash
docker compose build
docker compose up
```

Open the dashboard at `http://localhost:7860`.

## Hugging Face Spaces

This repo is ready for a free public Gradio Space. Set a write token in the environment, then deploy:

```bash
set HF_TOKEN=your_huggingface_write_token
set HF_SPACE_NAME=NeuroSymbolic-Meta-Reasoner
python deploy_to_hf.py
```

The Space runs `app.py` and installs the project from the root `requirements.txt`.

## Kaggle

A Kaggle kernel package can be staged and pushed with:

```bash
python scripts/publish_kaggle.py --stage
python scripts/publish_kaggle.py --push
```

Set `KAGGLE_USERNAME`, `KAGGLE_KEY`, and optionally `KAGGLE_KERNEL_ID=username/neurosymbolic-meta-reasoning-agent` before pushing.

## Tests

```bash
python -m pytest
```

## Project Layout

```text
neurosymbolic_agent/
  agent/              meta-controller, reasoning engine, self-improvement
  llm/                local LLM manager and model catalog
  memory/             working, episodic, and vector memory
  neural/             embeddings, classifier, neural inference
  symbolic/           Z3/SymPy/pattern-based symbolic solvers
  constitutional/     principles, checker, reward-hacking detector
  planning/           hierarchical planner and fallback strategies
  utils/              config, logging, telemetry, trace recorder
```
