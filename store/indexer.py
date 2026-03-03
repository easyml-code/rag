"""
indexer.py
----------
Public function: embed_and_store()

Takes chunks directly from PDFChunker and persists them to Chroma + SQLite.
"""

from typing import Any, Dict, List

from log.logs import logger
from utils.embeddings import clean_text, embed_documents
from store.vector_store import upsert as vector_upsert
from store.text_store import insert as text_insert


async def embed_and_store(chunks: List[Dict[str, Any]]) -> None:
    """
    Embed a list of chunk dicts and persist to Chroma + SQLite.

    Args:
        chunks: List of chunk dicts from PDFChunker.run().
                Must contain at minimum: chunk_id, chunk_text, doc_metadata.
    """
    logger.info("indexer start  chunks=%d", len(chunks))

    # 1. Clean texts for embedding (strips image refs and PPT artefacts)
    cleaned_texts = [clean_text(c.get("chunk_text", "")) for c in chunks]

    # 2. Embed
    vectors = await embed_documents(cleaned_texts)

    # 3. Persist to Chroma (vectors) and SQLite (full text + metadata)
    await vector_upsert(chunks, cleaned_texts, vectors)
    await text_insert(chunks, cleaned_texts)

    logger.info("indexer done  chunks=%d", len(chunks))