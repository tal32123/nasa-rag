import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and d.name.startswith("chroma_db")
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()

            for col in collections:
                key = f"{chroma_dir.name}::{col.name}"
                try:
                    doc_count = col.count()
                except Exception:
                    doc_count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": col.name,
                    "display_name": f"{chroma_dir.name} / {col.name} ({doc_count} docs)",
                    "doc_count": str(doc_count),
                }
        except Exception as exc:
            key = f"{chroma_dir.name}::error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {str(exc)[:50]})",
                "doc_count": "0",
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system; returns (collection, success, error_message)."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY", "")
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-3-small"
        )
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(
            name=collection_name, embedding_function=embedding_fn
        )
        return collection, True, ""
    except Exception as exc:
        return None, False, str(exc)

def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    where = (
        {"mission": mission_filter}
        if mission_filter and mission_filter.lower() != "all"
        else None
    )
    return collection.query(query_texts=[query], n_results=n_results, where=where)

def format_context(
    documents: List[str],
    metadatas: List[Dict],
    distances: Optional[List[float]] = None,
    ids: Optional[List[str]] = None,
) -> str:
    """Format retrieved documents into context, sorted by similarity and deduplicated."""
    if not documents:
        return ""

    have_scores = distances is not None and ids is not None
    items = list(zip(
        documents,
        metadatas,
        distances if have_scores else [None] * len(documents),
        ids if have_scores else [None] * len(documents),
    ))

    if have_scores:
        seen_ids: set = set()
        deduped = []
        for item in items:
            if item[3] not in seen_ids:
                seen_ids.add(item[3])
                deduped.append(item)
        items = sorted(deduped, key=lambda x: x[2])

    _MAX_CHARS = 1500
    parts = ["=== Retrieved NASA Mission Context ===\n"]

    for i, (doc, meta, dist, _) in enumerate(items, start=1):
        mission = meta.get("mission", "unknown").replace("_", " ").title()
        source = meta.get("source", "unknown")
        category = meta.get("document_category", "unknown").replace("_", " ").title()

        if dist is not None:
            header = f"[Source {i} | {mission} | {category} | {source} | Similarity: {1 - dist:.3f}]"
        else:
            header = f"[Source {i} | {mission} | {category} | {source}]"

        content = doc if len(doc) <= _MAX_CHARS else doc[:_MAX_CHARS] + "..."
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)