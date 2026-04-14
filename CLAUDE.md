# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Harry Potter lore agent (`lore_agent.ipynb`) that uses Google ADK for agent orchestration and ChromaDB as a vector store for RAG over HP lore content.

## Environment Setup

The virtualenv is at `./list/` (managed by `uv`, Python 3.8):

```bash
# Activate virtualenv
source list/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Running the Notebook

```bash
source list/bin/activate
jupyter notebook lore_agent.ipynb
```

## Key Dependencies

- **google-adk** (`0.0.1`) — Google Agent Development Kit; the agent framework driving the lore agent
- **chromadb** — vector database for storing and querying HP lore embeddings
- **huggingface-hub** + **tokenizers** + **onnxruntime** — local embedding model inference
- **fastapi** + **uvicorn** — HTTP server if the agent is exposed as an API
- **python-dotenv** — loads API keys / config from a `.env` file

## Architecture Notes

The intended architecture is a RAG pipeline:
1. HP lore text is embedded and stored in ChromaDB
2. At query time, relevant passages are retrieved from ChromaDB
3. Google ADK orchestrates an agent that uses the retrieved context to answer lore questions
