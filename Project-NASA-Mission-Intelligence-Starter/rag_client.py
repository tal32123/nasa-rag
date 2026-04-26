import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)

    # TODO: Loop through each discovered directory
        # TODO: Wrap connection attempt in try-except block for error handling
        
            # TODO: Initialize database client with directory path and configuration settings
            
            # TODO: Retrieve list of available collections from the database
            
            # TODO: Loop through each collection found
                # TODO: Create unique identifier key combining directory and collection names
                # TODO: Build information dictionary containing:
                    # TODO: Store directory path as string
                    # TODO: Store collection name
                    # TODO: Create user-friendly display name
                    # TODO: Get document count with fallback for unsupported operations
                # TODO: Add collection information to backends dictionary
        
        # TODO: Handle connection or access errors gracefully
            # TODO: Create fallback entry for inaccessible directories
            # TODO: Include error information in display name with truncation
            # TODO: Set appropriate fallback values for missing information

    # TODO: Return complete backends dictionary with all discovered collections

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO: Create a chomadb persistentclient
    # TODO: Return the collection with the collection_name

def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    where = (
        {"mission": mission_filter}
        if mission_filter and mission_filter.lower() != "all"
        else None
    )
    return collection.query(query_texts=[query], n_results=n_results, where=where)

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    _MAX_CHARS = 1500
    parts = ["=== Retrieved NASA Mission Context ===\n"]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        mission = meta.get("mission", "unknown").replace("_", " ").title()
        source = meta.get("source", "unknown")
        category = meta.get("document_category", "unknown").replace("_", " ").title()

        header = f"[Source {i} | {mission} | {category} | {source}]"
        content = doc if len(doc) <= _MAX_CHARS else doc[:_MAX_CHARS] + "..."
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)