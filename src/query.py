"""
Query the ingested documents: hybrid retrieval (BM25 JSON index + Chroma embeddings),
reciprocal-rank fusion to top-k, then an Ollama chat model aggregates the answer.

Usage:
    python -m src.query "what is BM25 indexing?"
    python -m src.query "explain the document processing pipeline" --top-k 8
    python -m src.query "what formats are supported?" --no-llm   # retrieval only (no LLM)
    python -m src.query "..." --model qwen2.5-coder:14b           # override default chat model
"""
import argparse
import os
import sys

import ollama

from src.core.bm25_index import BM25Index
from src.core.bm25_search import BM25Search
from src.core.hybrid_retriever import HybridRetriever
from src.core.query_processor import QueryProcessor
from src.core.retrieval_result import RetrievalResult
from src.core.vector_search import VectorSearch
from src.utils.database import VectorDatabase
from src.utils.log import get_logger

logger = get_logger("query")

BM25_INDEX_PATH = "data/embeddings/bm25_index.json"
CHROMA_PATH = "data/embeddings/chroma"
COLLECTION_NAME = "documents"
# Default chat model; override with --model or env OLLAMA_QUERY_MODEL
DEFAULT_LLM_MODEL = os.environ.get("OLLAMA_QUERY_MODEL", "deepseek-r1:8b")


def load_components() -> tuple[BM25Index, VectorDatabase, QueryProcessor]:
    logger.info("Loading BM25 index from %s", BM25_INDEX_PATH)
    index = BM25Index.load(BM25_INDEX_PATH)

    logger.info("Connecting to ChromaDB at %s", CHROMA_PATH)
    db = VectorDatabase(mode="dev", chroma_path=CHROMA_PATH)

    return index, db, QueryProcessor()


def retrieve(
    query_text: str,
    index: BM25Index,
    db: VectorDatabase,
    qp: QueryProcessor,
    top_k: int,
) -> list[dict]:
    processed = qp.process_query(query_text)
    bm25_query = " ".join(processed.all_terms)

    hybrid = HybridRetriever(
        BM25Search(index),
        VectorSearch(db, COLLECTION_NAME),
    )
    fused: list[RetrievalResult] = hybrid.retrieve(
        bm25_query,
        query_text,
        k=top_k,
        collection_name_for_cache=COLLECTION_NAME,
    )
    return [r.to_legacy_dict() for r in fused]


def generate_answer(query_text: str, context_chunks: list[dict], model: str) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['id']}]\n{c['text']}" for c in context_chunks
    )
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the ingested documents."

Context:
{context}

Question: {query_text}

Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def run_query(
    query_text: str,
    top_k: int = 5,
    use_llm: bool = True,
    llm_model: str | None = None,
) -> None:
    index, db, qp = load_components()

    processed = qp.process_query(query_text)
    print(f"\nQuery    : {query_text}")
    print(f"Intent   : {processed.intent.value}")
    print(f"Tokens   : {processed.tokens}")
    if processed.expanded_terms:
        print(f"Expanded : {processed.expanded_terms}")

    chunks = retrieve(query_text, index, db, qp, top_k=top_k)

    print(f"\n── Retrieved chunks ({len(chunks)}) ──────────────────────────────")
    for c in chunks:
        score_str = f"{c['score']:.4f}" if c['score'] else "n/a"
        print(f"  [{c['source']:6s} {score_str}] {c['id']}")
        print(f"    {c['text'][:120].strip()}...")

    if use_llm and chunks:
        model = llm_model or DEFAULT_LLM_MODEL
        print("\n── Answer ────────────────────────────────────────────────────")
        print(f"(model: {model})\n")
        answer = generate_answer(query_text, chunks, model=model)
        print(answer)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query ingested documents")
    parser.add_argument("query", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve (default: 5)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM, show retrieved chunks only")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama chat model (default: {DEFAULT_LLM_MODEL!r} or OLLAMA_QUERY_MODEL env)",
    )
    args = parser.parse_args()

    try:
        run_query(
            args.query,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            llm_model=args.model,
        )
    except FileNotFoundError:
        print("Error: BM25 index not found. Run ingestion first:", file=sys.stderr)
        print("  python -m src.ingest --docs data/documents", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
