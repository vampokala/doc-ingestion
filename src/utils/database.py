'''
- ChromaDB for development
- Qdrant for production scaling
- Embedding generation via Ollama
- Batch operations for efficiency
- Metadata filtering capabilities
'''
import logging
from typing import Dict, List, Optional

import chromadb
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
BATCH_SIZE = 100


class VectorDatabase:
    def __init__(self, mode: str = "dev", qdrant_host: str = "localhost", qdrant_port: int = 6333, chroma_path: str = "./chroma_db"):
        self.mode = mode
        self._chroma_path = chroma_path
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port
        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._qdrant_client: Optional[QdrantClient] = None

    # --- client accessors (lazy init) ---

    @property
    def chroma_client(self) -> chromadb.ClientAPI:
        if self._chroma_client is None:
            self._chroma_client = chromadb.PersistentClient(path=self._chroma_path)
            logger.info("ChromaDB initialized at %s", self._chroma_path)
        return self._chroma_client

    @property
    def qdrant_client(self) -> QdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(host=self._qdrant_host, port=self._qdrant_port)
            logger.info("Qdrant initialized at %s:%s", self._qdrant_host, self._qdrant_port)
        return self._qdrant_client

    # --- embedding ---

    def generate_embedding(self, text: str) -> List[float]:
        response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
        return response["embedding"]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.generate_embedding(t) for t in texts]

    # --- collection management ---

    def create_collection(self, collection_name: str) -> None:
        if self.mode == "dev":
            self.chroma_client.get_or_create_collection(name=collection_name)
            logger.info("ChromaDB collection %r ready", collection_name)
        else:
            if not self.qdrant_client.collection_exists(collection_name):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                logger.info("Qdrant collection %r created", collection_name)

    # --- insert ---

    def add_documents(self, collection_name: str, documents: List[Dict]) -> None:
        """Insert documents with embeddings generated from document['text'].

        Each document dict must have 'id' and 'text'; all other keys go into metadata.
        """
        for batch_start in range(0, len(documents), BATCH_SIZE):
            batch = documents[batch_start: batch_start + BATCH_SIZE]
            texts = [doc["text"] for doc in batch]
            embeddings = self.generate_embeddings_batch(texts)

            if self.mode == "dev":
                collection = self.chroma_client.get_or_create_collection(name=collection_name)
                collection.upsert(
                    ids=[str(doc["id"]) for doc in batch],
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=[{k: v for k, v in doc.items() if k not in ("id", "text")} for doc in batch],
                )
            else:
                points = [
                    PointStruct(
                        id=doc["id"],
                        vector=embedding,
                        payload={k: v for k, v in doc.items() if k not in ("id", "text")},
                    )
                    for doc, embedding in zip(batch, embeddings)
                ]
                self.qdrant_client.upsert(collection_name=collection_name, points=points)

            logger.info("Upserted batch of %d documents into %r", len(batch), collection_name)

    # --- query ---

    def query_documents(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search for similar documents, optionally filtered by metadata key/value pairs."""
        query_embedding = self.generate_embedding(query_text)

        if self.mode == "dev":
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            where = ({k: v for k, v in filters.items()} if filters else None)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
            return [
                {"id": id_, "text": doc, "metadata": meta, "distance": dist}
                for id_, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        else:
            search_filter = None
            if filters:
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
                search_filter = Filter(must=conditions)

            hits = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
            )
            return [
                {"id": hit.id, "metadata": hit.payload, "score": hit.score}
                for hit in hits
            ]
