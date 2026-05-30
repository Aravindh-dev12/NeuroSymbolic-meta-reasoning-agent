# NeuroSymbolic Agent Package

Core Python package for the NeuroSymbolic Meta-Reasoning Agent.

Use the installed CLI from the repository root:

```bash
python -m pip install -e ".[dashboard,dev]"
neuro-agent run "All mammals breathe air. Whales are mammals. Do whales breathe air?"
neuro-agent chat
neuro-agent models
neuro-agent doctor
```

Main components:

- `agent/`: meta-controller, reasoning engine, recursive self-improvement
- `llm/`: local model manager for Ollama, llama.cpp, Transformers, and offline fallback
- `memory/`: working memory, episodic memory, vector memory
- `neural/`: embeddings, classifier, neural inference
- `symbolic/`: Z3, SymPy, arithmetic, and syllogistic reasoning
- `constitutional/`: guardrails and reward-hacking detection
- `planning/`: hierarchical planning and fallback strategy selection
