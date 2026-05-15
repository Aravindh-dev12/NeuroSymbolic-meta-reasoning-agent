# NeuroSymbolic AGI Agent (Production-Grade)

> A production-grade meta-cognitive AGI agent that dynamically routes between neural and symbolic reasoning paths, with recursive self-improvement and constitutional safety constraints. **Now with local LLM support!**

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Meta-Controller                          │
│         (routes tasks, evaluates confidence, critiques)         │
│         Supports: Local (llama.cpp), Anthropic, OpenAI          │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌────────────────────┐
│  Neural Path  │    │  Symbolic Path     │
│  (PyTorch)    │    │  (Z3 / Prolog)     │
│  Embeddings   │    │  Logic Solver      │
│  Classifiers  │    │  Constraint Engine │
└───────┬───────┘    └────────┬───────────┘
        └──────────┬──────────┘
                   ▼
        ┌──────────────────────┐
        │   Working Memory     │
        │   (Chroma/FAISS/SQLite)│
        │   Vector Similarity  │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Constitutional AI   │
        │  Constraint Checker  │
        └──────────────────────┘
```

## 🚀 New Production Features

- **Local LLM Support**: Run with llama.cpp, Ollama, or cloud LLMs (Anthropic, OpenAI)
- **Vector Database**: ChromaDB, FAISS, or SQLite for scalable memory
- **Production Telemetry**: Built-in performance monitoring and resource tracking
- **Docker Support**: Containerized deployment with GPU support
- **Jupyter Integration**: Interactive notebook with widgets
- **Environment Configuration**: Full environment variable support
- **Automatic Model Download**: Downloads quantized models automatically

## Features

- **Dynamic Routing**: LLM controller selects neural vs. symbolic path per task type and confidence score
- **Recursive Self-Improvement**: Agent critiques outputs, detects failure modes, generates corrective traces
- **Working Memory**: Episodic memory + short-term working memory with vector similarity retrieval
- **Constitutional Constraints**: Rule-based guardrails to prevent reward hacking and unsafe outputs
- **Recursive Planning**: Hierarchical task decomposition with fallback strategies
- **Cross-Domain Generalisation**: No task-specific fine-tuning required

## Project Structure

```
neurosymbolic_agent/
├── agent/
│   ├── __init__.py
│   ├── meta_controller.py        # LLM meta-controller & router
│   ├── reasoning_engine.py       # Orchestrates neural + symbolic paths
│   └── self_improvement.py       # Recursive critique & correction loop
├── symbolic/
│   ├── __init__.py
│   ├── solver.py                 # Z3-based symbolic solver
│   ├── knowledge_base.py         # Prolog-style fact/rule store
│   └── constraint_engine.py      # Hard constraint enforcement
├── neural/
│   ├── __init__.py
│   ├── embedder.py               # Sentence/task embeddings
│   ├── classifier.py             # Task-type & confidence classifier
│   └── inference.py              # Neural inference pipeline
├── memory/
│   ├── __init__.py
│   ├── working_memory.py         # Short-term working memory
│   ├── episodic_memory.py        # Long-term episodic store
│   └── memory_manager.py         # Unified memory interface
├── constitutional/
│   ├── __init__.py
│   ├── principles.py             # Constitutional principles registry
│   ├── checker.py                # Output safety checker
│   └── reward_hacking_detector.py
├── planning/
│   ├── __init__.py
│   ├── hierarchical_planner.py   # Recursive task decomposition
│   ├── execution_engine.py       # Plan execution + monitoring
│   └── fallback_strategies.py    # Failure recovery
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── config.py
│   └── trace_recorder.py         # Reasoning trace recorder
├── tests/
│   ├── test_meta_controller.py
│   ├── test_symbolic_solver.py
│   ├── test_self_improvement.py
│   ├── test_constitutional.py
│   └── test_integration.py
├── configs/
│   ├── agent_config.yaml
│   └── constitutional_principles.yaml
├── notebooks/
│   └── demo.ipynb
├── main.py                       # Entry point
├── requirements.txt
└── README.md
```

## Quick Start

### Local LLM (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12.1)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CPU only
pip install llama-cpp-python

# Run with local LLM (auto-downloads model)
python main.py --task "Solve: All mammals breathe. Whales are mammals. Do whales breathe?"

# Interactive mode
python main.py --interactive
```

### Cloud LLMs

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your_key_here
# or
export OPENAI_API_KEY=your_key_here

# Run with cloud LLM
python main.py --backend anthropic --task "Classify this sentiment: The product exceeded all expectations"
```

### Docker

```bash
# Build and run
docker-compose up -d

# Interactive mode
docker-compose run neurosymbolic-agent python main.py --interactive
```

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/interactive_agent.ipynb
```

## Architecture Deep-Dive

### Meta-Controller
The LLM controller (Claude) analyses each task and decides:
1. **Path Selection**: Neural (pattern-matching, classification, NLU) vs Symbolic (logic, constraints, formal verification)
2. **Confidence Estimation**: If confidence < threshold, triggers self-improvement loop
3. **Memory Query**: Retrieves relevant past episodes before reasoning

### Self-Improvement Loop
```
Output → Critique → Failure Detection → Corrective Trace → Memory Update → Re-attempt
```
Constitutional constraints are checked at every step to prevent reward hacking.

### Constitutional Constraints
Defined in `configs/constitutional_principles.yaml`, enforced at runtime:
- Factual grounding (no hallucination without uncertainty markers)
- No circular reasoning loops
- Confidence must be calibrated (no overconfidence)
- No self-modification of safety constraints

## Configuration

### Environment Variables

Configure the agent using environment variables (recommended for production):

```bash
# LLM Backend
export LLM_BACKEND=local  # Options: local, anthropic, openai
export LOCAL_MODEL_NAME=llama3-8b  # For local backend
export ANTHROPIC_API_KEY=your_key  # For Anthropic
export OPENAI_API_KEY=your_key  # For OpenAI

# Agent Configuration
export CONFIDENCE_THRESHOLD=0.75
export ENABLE_TELEMETRY=true
export VERBOSE=true

# Memory Configuration
export VECTOR_DB_TYPE=chroma  # Options: chroma, faiss, sqlite
export VECTOR_DB_DIR=data/vector_db

# Logging
export LOG_LEVEL=INFO
export LOG_TO_CONSOLE=true
```

### Configuration File

Or edit `configs/agent_config.yaml`:
```yaml
agent:
  name: NeuroSymbolicMetaAgent
  version: 2.0.0
  llm_backend: local  # local, anthropic, openai
  local_model_name: llama3-8b
  confidence_threshold: 0.75
  max_self_improvement_rounds: 3
  max_planning_depth: 5
  enable_telemetry: true

memory:
  vector_db_type: chroma  # chroma, faiss, sqlite
  persist_directory: data/vector_db
  retrieval_top_k: 5
  similarity_threshold: 0.6

routing:
  symbolic_task_types: [logic, math, constraint, formal]
  neural_task_types: [nlp, classification, sentiment, generation]
```

## Available Local Models

The agent supports these pre-configured local models:

- **llama3-8b**: Meta Llama 3 8B (4-bit quantized, ~4.7GB)
- **llama3-70b**: Meta Llama 3 70B (4-bit quantized, ~39GB)
- **mistral-7b**: Mistral 7B Instruct (4-bit quantized, ~4.3GB)
- **mixtral-8x7b**: Mixtral 8x7B (4-bit quantized, ~26GB)
- **ollama-llama3**: Use Ollama backend (requires `ollama serve`)

Models are automatically downloaded on first use from HuggingFace.

## Deployment

For production deployment, see [DEPLOYMENT.md](../DEPLOYMENT.md) for:
- Docker deployment
- Kubernetes configuration
- Production monitoring
- Security best practices
- Scaling strategies

## Monitoring & Telemetry

The agent includes built-in telemetry:

```bash
# View telemetry logs
cat logs/telemetry.jsonl | jq

# View agent logs
tail -f logs/agent.log

# View reasoning traces
tail -f logs/reasoning_traces.jsonl
```

Telemetry tracks:
- Task duration per stage
- Resource usage (CPU, memory)
- Routing decisions
- Self-improvement rounds
- Constitutional violations
- Success/failure rates

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=neurosymbolic_agent --cov-report=html

# Run specific test
pytest tests/test_meta_controller.py
```

### Code Structure

```
neurosymbolic_agent/
├── agent/              # Core agent logic
│   ├── meta_controller.py      # LLM routing with multi-backend support
│   ├── reasoning_engine.py     # Neural/symbolic execution
│   └── self_improvement.py     # Recursive critique loop
├── llm/                # Local LLM integration
│   └── local_llm_manager.py    # llama.cpp, Ollama, Transformers
├── memory/             # Memory systems
│   ├── vector_memory.py        # Chroma/FAISS/SQLite vector DB
│   ├── memory_manager.py       # Unified memory interface
│   ├── episodic_memory.py      # Long-term episodic storage
│   └── working_memory.py       # Short-term working memory
├── neural/             # Neural components
│   ├── embedder.py             # Sentence embeddings
│   ├── classifier.py           # Task classification
│   └── inference.py            # Neural inference
├── symbolic/           # Symbolic reasoning
│   ├── solver.py               # Z3 constraint solver
│   ├── knowledge_base.py       # Prolog-style facts/rules
│   └── constraint_engine.py    # Hard constraint enforcement
├── constitutional/     # Safety constraints
│   ├── checker.py              # Output validation
│   ├── principles.py           # Constitutional principles
│   └── reward_hacking_detector.py
├── planning/           # Task planning
│   ├── hierarchical_planner.py # Task decomposition
│   └── fallback_strategies.py  # Failure recovery
└── utils/              # Utilities
    ├── config.py               # Configuration with env vars
    ├── telemetry.py            # Performance monitoring
    ├── logging_utils.py        # Structured logging
    └── trace_recorder.py       # Reasoning traces
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Meta for Llama 3 models
- Mistral AI for Mistral models
- llama.cpp for efficient local inference
- Anthropic for Claude API
- OpenAI for GPT models
- The open-source AI community
