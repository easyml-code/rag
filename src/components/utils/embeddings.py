"""
embeddings.py
-------------
Responsible for:
  - Cleaning chunk text (strip image refs) before embedding
  - Async batch embedding via Google text-embedding-004
"""

import re
import asyncio
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings


_IMAGE_MD_RE  = re.compile(r'!\[.*?\]\([^)]*\)')    # ![alt](file.png)
_HTML_IMG_RE  = re.compile(r'<img[^>]+>',  re.IGNORECASE)
_HTML_DIV_RE  = re.compile(r'<div[^>]*>|</div>', re.IGNORECASE)
_MULTI_NL_RE  = re.compile(r'\n{3,}')


def clean_text(text: str) -> str:
    """Strip image refs and collapse whitespace. Used before embedding."""
    text = _IMAGE_MD_RE.sub('', text)
    text = _HTML_IMG_RE.sub('', text)
    text = _HTML_DIV_RE.sub('', text)
    text = _MULTI_NL_RE.sub('\n\n', text)
    return text.strip()


def _get_embedder(task_type: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        task_type=task_type,
    )


async def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Batch-embed a list of document strings.
    Runs Google API calls in a thread (non-blocking).
    """
    embedder = _get_embedder("RETRIEVAL_DOCUMENT")
    all_vectors: List[List[float]] = []
    total_batches = (len(texts) + settings.embed_batch_size - 1) // settings.embed_batch_size

    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        batch_num = i // settings.embed_batch_size + 1
        print(f"  [embeddings] batch {batch_num}/{total_batches} — {len(batch)} texts")
        vectors = await asyncio.to_thread(embedder.embed_documents, batch)
        all_vectors.extend(vectors)

    return all_vectors


async def embed_query(query: str) -> List[float]:
    """Embed a single user query string."""
    embedder = _get_embedder("RETRIEVAL_QUERY")
    return await asyncio.to_thread(embedder.embed_query, query)
