"""
vector_store.py
---------------
Chroma vector DB operations:
  - upsert : store embeddings + lightweight metadata
  - query  : ANN search, returns top-k with cosine similarity scores
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from src.config import settings


def _collection() -> chromadb.Collection:
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Build lightweight metadata stored in Chroma alongside each vector."""
    doc_meta = chunk.get("doc_metadata", {})
    bbox = chunk.get("bbox")
    return {
        "doc_id":                doc_meta.get("doc_id", ""),
        "page":                  int(chunk.get("page", 0)),
        "chunk_type":            chunk.get("chunk_type", ""),
        "image_path":            chunk.get("image", ""),   # relative to settings.images_dir
        "section_title":         chunk.get("section_title") or "",
        "token_count":           int(chunk.get("token_count", 0)),
        "extraction_confidence": chunk.get("extraction_confidence", ""),
        "image_width_px":        int(chunk.get("image_width_px", 0)),
        "image_height_px":       int(chunk.get("image_height_px", 0)),
        "bbox_json":             json.dumps(bbox) if bbox is not None else "",
        "source_uri":            doc_meta.get("source_uri", ""),
        "source_file":           doc_meta.get("source_file", ""),
        "filename":              doc_meta.get("filename", ""),   # original upload name
        "total_pages":           int(doc_meta.get("total_pages", 0)),
        "pdf_type":              doc_meta.get("pdf_type", ""),
        "created_at":            doc_meta.get("created_at", ""),
        "doc_metadata_json":     json.dumps(doc_meta),
    }


async def upsert(
    chunks: List[Dict[str, Any]],
    cleaned_texts: List[str],
    vectors: List[List[float]],
) -> None:
    """Upsert chunk embeddings into Chroma. Runs in thread (non-blocking)."""
    col = _collection()

    ids       = [c["chunk_id"] for c in chunks]
    metadatas = [_build_metadata(c) for c in chunks]

    def _upsert():
        for i in range(0, len(ids), 100):
            col.upsert(
                ids        = ids[i:i+100],
                embeddings = vectors[i:i+100],
                documents  = cleaned_texts[i:i+100],
                metadatas  = metadatas[i:i+100],
            )

    await asyncio.to_thread(_upsert)
    print(f"  [vector_store] upserted {len(ids)} vectors")


async def query(
    query_vector: List[float],
    top_k: int,
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query Chroma for top_k nearest chunks. Returns list with similarity scores."""
    col = _collection()
    where = {"doc_id": {"$eq": doc_id}} if doc_id else None

    def _query():
        return col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    raw = await asyncio.to_thread(_query)

    results = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]

        bbox = None
        bbox_json = meta.get("bbox_json") or ""
        if bbox_json:
            try:
                bbox = json.loads(bbox_json)
            except json.JSONDecodeError:
                pass

        doc_metadata: Dict[str, Any] = {}
        doc_meta_json = meta.get("doc_metadata_json") or ""
        if doc_meta_json:
            try:
                doc_metadata = json.loads(doc_meta_json)
            except json.JSONDecodeError:
                pass

        results.append({
            "chunk_id":              raw["ids"][0][i],
            "text":                  raw["documents"][0][i],
            "score":                 round(1.0 - raw["distances"][0][i], 6),
            "doc_id":                meta.get("doc_id"),
            "page":                  meta.get("page"),
            "chunk_type":            meta.get("chunk_type"),
            "image_path":            meta.get("image_path"),
            "section_title":         meta.get("section_title"),
            "token_count":           meta.get("token_count"),
            "extraction_confidence": meta.get("extraction_confidence"),
            "image_width_px":        meta.get("image_width_px"),
            "image_height_px":       meta.get("image_height_px"),
            "bbox":                  bbox,
            "source_uri":            meta.get("source_uri"),
            "source_file":           meta.get("source_file"),
            "filename":              meta.get("filename", ""),   # original upload name
            "total_pages":           meta.get("total_pages"),
            "pdf_type":              meta.get("pdf_type"),
            "created_at":            meta.get("created_at"),
            "doc_metadata":          doc_metadata,
            "retrieval_type":        "vector",
        })
    return results