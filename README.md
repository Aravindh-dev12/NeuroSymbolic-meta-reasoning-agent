# NeuroSymbolic AGI Agent (Production-Grade)

> A production-grade meta-cognitive AGI agent that dynamically routes between neural and symbolic reasoning paths, with recursive self-improvement and constitutional safety constraints. **Now with local LLM support!**

## 🚀 Features

- **Local LLM Support**: Run with llama.cpp, Ollama, or cloud LLMs (Anthropic, OpenAI)
- **Vector Database**: ChromaDB, FAISS, or SQLite for scalable memory
- **Production Telemetry**: Built-in performance monitoring and resource tracking
- **Docker Support**: Containerized deployment with GPU support
- **Jupyter Integration**: Interactive notebook with widgets
- **Environment Configuration**: Full environment variable support
- **Automatic Model Download**: Downloads quantized models automatically

## Quick Start

### Local LLM (Recommended)

```bash
# Install dependencies
pip install -r neurosymbolic_agent/requirements.txt

# For GPU support (CUDA 12.1)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CPU only
pip install llama-cpp-python

# Run with local LLM (auto-downloads model)
python main.py --task "Solve: All mammals breathe. Whales are mammals. Do whales breathe?"

# Interactive mode
python main.py --interactive
```

### Docker

```bash
# Build and run
docker-compose up -d

# Interactive mode
docker-compose run neurosymbolic-agent python main.py --interactive
```

## Documentation

For full documentation, see [neurosymbolic_agent/README.md](neurosymbolic_agent/README.md)

## Deployment

For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md)

## License

MIT License
