# Deployment Guide

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd neurosymbolic_agent

# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f neurosymbolic-agent

# Stop the service
docker-compose down
```

### Manual Docker Build

#### GPU Support (CUDA)

```bash
# Build with GPU support
docker build --target base -t neurosymbolic-agent:gpu .

# Run with GPU
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  -e LLM_BACKEND=local \
  -e LOCAL_MODEL_NAME=llama3-8b \
  neurosymbolic-agent:gpu
```

#### CPU Only

```bash
# Build CPU version
docker build --target cpu-base -t neurosymbolic-agent:cpu .

# Run CPU version
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  -e LLM_BACKEND=local \
  -e LOCAL_MODEL_NAME=llama3-8b \
  neurosymbolic-agent:cpu
```

### Environment Variables

Configure the agent using environment variables:

```bash
# LLM Backend
LLM_BACKEND=local  # Options: local, anthropic, openai
LOCAL_MODEL_NAME=llama3-8b  # For local backend
ANTHROPIC_API_KEY=your_key_here  # For Anthropic backend
OPENAI_API_KEY=your_key_here  # For OpenAI backend

# Agent Configuration
CONFIDENCE_THRESHOLD=0.75
ENABLE_TELEMETRY=true
VERBOSE=true

# Memory Configuration
VECTOR_DB_TYPE=chroma  # Options: chroma, faiss, sqlite
VECTOR_DB_DIR=/app/data/vector_db

# Logging
LOG_LEVEL=INFO
LOG_TO_CONSOLE=true
```

### Jupyter Notebook with Docker

```bash
# Start Jupyter service
docker-compose up jupyter

# Access notebook at http://localhost:8888
```

## Local Deployment

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- 16GB+ RAM recommended
- 20GB+ disk space for models

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12.1)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# For CPU only
pip install llama-cpp-python
```

### Configuration

Create `.env` file:

```bash
LLM_BACKEND=local
LOCAL_MODEL_NAME=llama3-8b
CONFIDENCE_THRESHOLD=0.75
ENABLE_TELEMETRY=true
VECTOR_DB_TYPE=chroma
LOG_LEVEL=INFO
```

### Running the Agent

```bash
# Interactive mode
python main.py --interactive

# Single task
python main.py --task "Your task here"

# With specific backend
python main.py --backend anthropic --task "Your task here"

# With specific model
python main.py --backend local --model mistral-7b --task "Your task here"
```

## Production Deployment

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB SSD
- GPU: Optional (recommended for local LLMs)

**Recommended:**
- CPU: 8+ cores
- RAM: 32GB+
- Storage: 100GB+ NVMe SSD
- GPU: NVIDIA RTX 3090/4090 or equivalent with 24GB+ VRAM

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up persistent storage (data, logs, models)
- [ ] Configure logging and monitoring
- [ ] Set up backup strategy
- [ ] Configure firewall/security
- [ ] Set up resource limits
- [ ] Configure health checks
- [ ] Set up log rotation
- [ ] Test failover procedures

### Monitoring

The agent includes built-in telemetry:

```bash
# View telemetry logs
cat logs/telemetry.jsonl | jq

# View agent logs
tail -f logs/agent.log

# View reasoning traces
tail -f logs/reasoning_traces.jsonl
```

### Scaling

For horizontal scaling:

1. Use a shared vector database (ChromaDB with persistent storage)
2. Configure shared storage for episodic memory
3. Use load balancer for multiple instances
4. Consider using a message queue for task distribution

### Security

- Never commit API keys to version control
- Use environment variables or secret management
- Enable constitutional constraints
- Regular security audits
- Network isolation for production deployments
- Rate limiting for API endpoints

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce model size (use 8B instead of 70B)
- Increase system swap
- Use quantization (q4_k_m)
- Reduce context length

**Slow Performance:**
- Enable GPU acceleration
- Use FAISS instead of ChromaDB for faster similarity search
- Reduce memory retrieval top_k
- Cache embeddings

**Model Download Fails:**
- Check internet connection
- Verify model URL in local_llm_manager.py
- Manually download and place in models/ directory
- Use Ollama backend as alternative

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
export VERBOSE=true

# Disable telemetry
export ENABLE_TELEMETRY=false

# Run with debug output
python main.py --task "test" --verbose
```
