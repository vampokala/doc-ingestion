---
title: Doc Ingestion RAG Demo
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.37.0"
app_file: spaces/app.py
pinned: false
license: mit
short_description: Citation-aware RAG with hybrid retrieval and truthfulness scoring
---

# Doc Ingestion — RAG Demo

A live demo of **Doc-Ingestion**, a citation-aware Retrieval-Augmented Generation system.

## What you can do here

- Ask questions about **RAG**, **vector databases**, and **BM25 retrieval** — sample documents are pre-loaded.
- See **citations** pointing to the specific chunks that grounded each answer.
- See a **truthfulness score** (NLI faithfulness + citation groundedness) for every response.
- Choose between **OpenAI**, **Anthropic**, or **Gemini** providers using your own key (paste it in the sidebar).

## Limitations of this demo

- **Uploads are disabled** — this demo runs on pre-ingested sample documents only.
- **No persistence** — embeddings are stored in-memory and reset on each Space restart.
- **Cloud LLM only** — Ollama (local model) is not available in this hosted environment.

## Run locally with full features

```bash
git clone https://github.com/vampokala/Doc-Ingestion
cd Doc-Ingestion
bash scripts/bootstrap_demo.sh
```

Or with Docker (one command):

```bash
cp docker/.env.example docker/.env
# Edit docker/.env to add your API keys
docker compose -f docker/docker-compose.yml up
```

Open http://localhost:8501 for the UI.

## Source code

[github.com/vampokala/Doc-Ingestion](https://github.com/vampokala/Doc-Ingestion)
