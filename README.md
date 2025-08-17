[![CI](https://github.com/ZeleMate/Agentic-RAG-PoC/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/ZeleMate/Agentic-RAG-PoC/actions/workflows/ci.yaml)

## Agentic RAG – Proof of Concept (LangGraph + Hugging Face + FAISS + Ollama)

This repository contains a minimal, local Agentic RAG (Retrieval‑Augmented Generation) proof of concept that demonstrates autonomous tool‑use (retrieve vs. respond), relevance grading, question rewriting, and grounded answer generation using LangGraph.

**Inspiration**: This implementation is inspired by the [Self-Reflective RAG with LangGraph](https://blog.langchain.com/agentic-rag-with-langgraph/) blog post, particularly the Self-RAG framework concepts. The notebook demonstrates key ideas from the paper including:
- **Retrieve** decision: autonomous choice between direct response or document retrieval
- **Relevancy** (relevance grading): binary assessment of retrieved document relevance
- **Query rewriting**: reformulating questions when retrieved context is irrelevant
- **Grounded generation**: producing answers faithful to retrieved context

### Architecture

image.png

### Reproducibility and Setup

Requirements
- Python >= 3.11
- macOS/Linux/Windows with ability to run local LLM via Ollama
- No paid API keys required

1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2) Install project dependencies
- Using `pip` directly (matches the notebook):
```bash
pip install -U \
  transformers \
  langgraph langchain-community langchain-text-splitters langchain-ollama \
  sentence-transformers \
  torch langchain-docling \
  faiss-cpu
```

3) Install and prepare Ollama (for the local LLM)
- Install Ollama: see `https://ollama.com`
- Pull a compatible model used in the notebook (example below). Adjust if you prefer a different local model.
```bash
ollama pull qwen3:4b
```

4) Data source
- The notebook fetches a single arXiv PDF directly:
  - `https://arxiv.org/pdf/2506.19676`

5) Run the notebook
```bash
jupyter lab  # or: jupyter notebook
# open notebooks/agentic_rag.ipynb and execute cells top-to-bottom
```

### Notebook Contents (What it Demonstrates)
1. Environment setup and minimal imports for reproducibility
2. Document loading and chunking (Docling hybrid chunker or markdown splitting)
3. Embedding with Sentence Transformers and FAISS vector index build
4. Retriever as a LangChain tool exposed to the agent
5. **Retrieve decision**: autonomous choice between direct response or document retrieval (inspired by Self-RAG's `Retrieve` token)
6. **Relevancy (relevance grading)**: LLM‑based binary assessment of retrieved document relevance
7. **Query rewriting**: reformulating questions when retrieved context is irrelevant (Self-RAG's self-correction mechanism)
8. **Grounded generation**: producing answers faithful to retrieved context (similar to Self-RAG's `ISSUP` verification)
9. LangGraph workflow assembly, visualization, and streaming run
10. Local ranking quality metric (NDCG@k) without extra dependencies

### How to Evaluate (Local NDCG@k)
The notebook includes a lightweight NDCG@k implementation using the existing grader LLM to assign binary relevance to retrieved chunks. Example output:
```
NDCG@5 report: {"k": 5, "num_queries": 2, "ndcg@k_mean": 0.50, "per_query": [0.0, 1.0]}
```
Recommendations:
- Use at least 25–50 queries for meaningful averages
- Tune chunk size/overlap, retrieval `k`, and try alternative embedding models

### Current Bottlenecks
- Local LLM latency (generation and grading) may be slow on CPU; use GPU where possible
- FAISS recall depends on chunking quality and embedding choice
- Small evaluation sets yield noisy metrics (increase query count)
- No re‑ranking step (cross‑encoder) in the minimal PoC

### Suggested Improvements and Scaling Paths
- Retrieval quality
  - Experiment with chunk size/overlap (e.g., 700–900 tokens, 10–20% overlap)
  - Try alternative embeddings (MiniLM L6/L12, BGE small) and hybrid retrieval (BM25 + dense)
  - Add optional cross‑encoder re‑ranking for top‑k results
- Agent loop quality
  - Strengthen rewrite and grading prompts (already prepared in notebook)
  - Log and compare retrieval before/after rewriting
- Performance
  - Cache retrieval and LLM calls for repeated queries
  - Batch processing for multiple queries; consider async execution
- Evaluation
  - Increase query set size; stratify by question type
  - Optionally add RAGAS metrics (faithfulness, answer relevancy) reusing the local LLM
- Scaling
  - Containerize components (vector DB, LLM, orchestrator)
  - Consider distributed vector stores and load‑balanced LLM backends
  - Introduce message queues for asynchronous pipelines

### Project Structure
```
Agentic-RAG-PoC/
  ├─ notebooks/
  │   └─ agentic_rag.ipynb        # end‑to‑end PoC with LangGraph
  ├─ data/                        # (optional) local files if you add your own corpus
  ├─ pyproject.toml               # pinned dependencies for reproducibility
  └─ README.md                    # this document
```

### Notes
- The PoC intentionally avoids paid APIs; everything is local and replaceable.
- The architecture and prompts are kept simple to be easily explained line‑by‑line.
- **Academic inspiration**: This implementation adapts concepts from the Self-RAG paper (Asai et al., 2023) and the LangChain blog post on [Self-Reflective RAG with LangGraph](https://blog.langchain.com/agentic-rag-with-langgraph/), demonstrating how state machines can enable self-correcting RAG workflows.
