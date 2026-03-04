"""
retriever.py
------------
Public function: retrieve()

Combines vector (Chroma) and BM25 text (SQLite) search.
"""

from typing import Any, Dict, List

from src.log.logs import logger
from src.components.utils.embeddings import embed_query
from src.components.ingestion.store.vector_store import query as vector_query
from src.components.ingestion.store.text_store import query as text_query
from src.components.retriever.image_loader import attach_images


async def retrieve(
    user_query: str,
    top_k: int = 5,
    images: bool = True,
    image_payload: str = "ref",
    text_search: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve top-k matching chunks for a query.

    Args:
        user_query:    Natural language query string.
        top_k:         Number of results per retrieval type.
        images:        If True, attach image data to each result.
        image_payload: "ref" → lightweight refs only.
                       "blob" → full raw bytes for multimodal model calls.
        text_search:   If True, run BOTH vector + BM25 text search.
                       If False, run ONLY vector search.

    Returns:
        {
            "vector_results": [ { chunk_id, text, score, page, chunk_type,
                                  image_path, images?, ... } ],
            "text_results":   [ ... ]   # only when text_search=True
        }
    """
    if images and image_payload not in {"ref", "blob"}:
        raise ValueError("image_payload must be 'ref' or 'blob'")

    print(f"[retriever] query={user_query!r}  top_k={top_k}  "
          f"images={images}  image_payload={image_payload}  text_search={text_search}")

    # 1. Embed query
    query_vector = await embed_query(user_query)

    # 2. Vector search (always runs)
    vector_results = await vector_query(query_vector, top_k)
    if images:
        vector_results = [attach_images(r, mode=image_payload) for r in vector_results]

    output: Dict[str, List[Dict[str, Any]]] = {"vector_results": vector_results}

    # 3. BM25 text search (optional)
    if text_search:
        try:
            bm25_results = await text_query(user_query, top_k)
            if images:
                bm25_results = [attach_images(r, mode=image_payload) for r in bm25_results]
            output["text_results"] = bm25_results
        except Exception as exc:
            # Keep chat/retrieve alive even if FTS tables are missing/corrupt.
            logger.warning("BM25 text_query failed, returning vector-only results: %s", exc)
            output["text_results"] = []

    return output
