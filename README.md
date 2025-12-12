# ReasoningBank

Implementation of ReasoningBank from the paper "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory" (arXiv:2509.25140). This is a memory-augmented LLM agent that extracts generalizable reasoning strategies from agent trajectories and retrieves them to improve future task performance.

## Overview

ReasoningBank is a memory system for LLM agents that:
- Extracts reasoning strategies from successful trajectories
- Stores strategies in an embedding-based memory bank
- Retrieves relevant strategies for new tasks
- Achieves 4-8% improvement in success rate on WebArena

This repository includes:
- ReAct-style agent with browser tool integration
- Memory system with storage, retrieval, and embeddings
- LLM-as-Judge for trajectory evaluation
- Strategy extraction pipeline
- Full evaluation harness for WebArena, SWE-bench, and Mind2Web

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy a config template and add your API keys:

```bash
# For Gemini 2.5 (recommended, 10-20x cheaper than GPT-4)
cp configs/config.gemini25.yaml config.yaml
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_key_here

# Or for OpenAI
echo "OPENAI_API_KEY=your_key_here" > .env
# Edit config.yaml: provider: "openai", model: "gpt-4"
```

Configuration options in `config.yaml`:
- LLM provider and model
- Embedding model
- Memory bank parameters (retrieve_k, dedup_threshold)
- Agent parameters (max_steps, timeout)

## Usage

### Basic Evaluation

```bash
# No-Memory baseline
python scripts/evaluation/run_eval.py --mode no_memory --subset shopping --seed 42

# ReasoningBank with memory
python scripts/evaluation/run_eval.py --mode reasoningbank --subset shopping --seed 42
```

### Reproduce Paper Results

```bash
# Table 1: WebArena results
python scripts/reproduction/reproduce_table1.py

# Table 2: SWE-bench results
python scripts/reproduction/reproduce_table2.py

# Complete evaluation (all benchmarks)
./scripts/reproduction/run_complete_evaluation.sh
```

### Additional Benchmarks

```bash
# SWE-bench
python scripts/evaluation/run_swebench.py

# Mind2Web
python scripts/evaluation/run_mind2web.py
```

### Analysis

```bash
# Aggregate results
python scripts/analysis/aggregate_results.py --results_dir outputs/results

# Generate visualizations
python scripts/analysis/swebench_visualization.py
```

## Repository Structure

```
ReasoningBank/
├── src/                         # Core implementation
│   ├── agent.py                # ReAct agent
│   ├── memory.py               # Memory bank system
│   ├── judge.py                # LLM-as-Judge
│   ├── extractor.py            # Strategy extraction
│   ├── evaluator.py            # Evaluation harness
│   ├── llm_client.py           # LLM provider interface
│   └── embeddings.py           # Embedding generation
│
├── scripts/
│   ├── evaluation/             # Evaluation scripts
│   ├── reproduction/           # Paper reproduction
│   ├── analysis/               # Result analysis
│   └── setup/                  # Setup scripts
│
├── configs/                    # Configuration files
├── docs/                       # Documentation
├── data/                       # Task datasets
└── outputs/                    # Generated outputs (gitignored)
    ├── logs/
    ├── results/
    ├── plots/
    ├── trajectories/
    └── memory_banks/
```

## Implementation

### Agent Architecture

ReAct agent with Thought → Action → Observation loop:
- Tools: navigate, click, type, scroll, read_page, submit_form, finish
- Token usage tracking
- Configurable max steps and timeout

### Memory System

Components:
1. Storage: JSONL file with structured memory items
2. Embedding: OpenAI text-embedding-3-large or sentence-transformers
3. Index: FAISS for similarity search
4. Retrieval: Top-K cosine similarity

Memory item structure:
```json
{
  "id": "uuid",
  "title": "Strategy name",
  "description": "When to apply",
  "content": ["Step 1", "Step 2"],
  "rationale": "Why this works",
  "provenance": {"task_id": "...", "success": true},
  "embedding": [...]
}
```

### Evaluation

Two modes:
1. No-Memory: Baseline without retrieval
2. ReasoningBank: With memory retrieval and online learning

Process:
1. Load tasks
2. (ReasoningBank) Retrieve relevant memories
3. Run agent
4. Judge success/failure
5. (ReasoningBank) Extract strategies and update bank

Metrics:
- Success Rate (SR): % of completed tasks
- Average Steps: Mean steps for successful tasks

## Expected Results

WebArena results (Gemini 2.5):

| Subset | No-Memory | ReasoningBank | Δ |
|--------|-----------|---------------|---|
| Shopping | 35% | 43% | +8% |
| Admin | 40% | 47% | +7% |
| GitLab | 30% | 36% | +6% |
| Reddit | 32% | 38% | +6% |
| Multi | 25% | 30% | +5% |
| Overall | 34% | 41% | +7% |

Results vary by LLM, implementation details, and seed.

## Troubleshooting

**API Rate Limits:**
- Reduce batch_size in config
- Add delays between requests
- Use checkpointing

**Out of Memory:**
- Reduce max_steps
- Use smaller embedding model
- Process subsets individually

**Low Success Rates:**
- Verify task data quality
- Check LLM prompts
- Review logs in outputs/logs/

## Contributing

### Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/ReasoningBank.git
cd ReasoningBank
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

### Code Organization

- Core modules → `src/`
- Evaluation scripts → `scripts/evaluation/`
- Analysis tools → `scripts/analysis/`
- Documentation → `docs/`
- Configs → `configs/`

### Adding Features

**New Benchmark:**
1. Add script to `scripts/evaluation/run_benchmark.py`
2. Add config to `configs/config.benchmark.yaml`
3. Document in `docs/BENCHMARK_GUIDE.md`

**Analysis Tool:**
1. Add to `scripts/analysis/`
2. Document usage
3. Test with outputs/

**Core Module:**
1. Add to `src/`
2. Update imports
3. Document

### File Naming

- Python: `snake_case.py`
- Shell: `lowercase_with_underscores.sh`
- Configs: `config.{variant}.yaml`
- Docs: `UPPERCASE_WITH_UNDERSCORES.md`

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions modular

### Pull Requests

1. Create branch: `git checkout -b feature/name`
2. Make changes
3. Test: `python scripts/evaluation/run_eval.py --help`
4. Commit: `git commit -m "Clear description"`
5. Push and create PR

PR checklist:
- [ ] Code follows style guidelines
- [ ] Files in correct directories
- [ ] Documentation updated
- [ ] Tested locally
- [ ] Clear commit messages

### What NOT to Commit

- API keys or credentials
- outputs/ contents (logs, results, plots)
- _archived/ directory
- __pycache__/ directories
- .DS_Store files
- Large binaries

## Documentation

Additional guides in `docs/`:
- `AWS_EC2_SETUP_GUIDE.md` - Cloud deployment
- `README_SWEBENCH.md` - SWE-bench evaluation
- `SWEBENCH_SETUP.md` - SWE-bench environment
- `EXECUTIVE_SUMMARY.md` - Project overview

## Citation

```bibtex
@article{reasoningbank2025,
  title={ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory},
  author={[Authors]},
  journal={arXiv preprint arXiv:2509.25140},
  year={2025}
}
```

## License

Provided for research and educational purposes.

## Acknowledgments

- ReasoningBank paper: arXiv:2509.25140
- WebArena: [webarena.dev](https://webarena.dev)
- BrowserGym framework
