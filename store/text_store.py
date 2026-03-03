"""
text_store.py
-------------
SQLite + FTS5 operations:
  - insert : store full chunk text and metadata, FTS5 indexed for BM25
  - query  : keyword search using FTS5 BM25 ranking
"""

import asyncio
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional

from config import settings


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id              TEXT PRIMARY KEY,
    doc_id                TEXT,
    chunk_type            TEXT,
    chunk_text            TEXT,
    cleaned_text          TEXT,
    page                  INTEGER,
    image_path            TEXT,
    section_title         TEXT,
    token_count           INTEGER,
    extraction_confidence TEXT,
    source_file           TEXT,
    doc_metadata          TEXT
);
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(cleaned_text, content=chunks, content_rowid=rowid);
"""

_CREATE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, cleaned_text) VALUES ('delete', old.rowid, old.cleaned_text);
    INSERT INTO chunks_fts(rowid, cleaned_text) VALUES (new.rowid, new.cleaned_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, cleaned_text) VALUES ('delete', old.rowid, old.cleaned_text);
END;
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id);"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(settings.sqlite_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _init(con: sqlite3.Connection) -> None:
    con.execute(_CREATE_TABLE)
    con.execute(_CREATE_FTS)
    con.executescript(_CREATE_TRIGGERS)
    con.execute(_CREATE_INDEX)
    con.commit()


async def insert(chunks: List[Dict[str, Any]], cleaned_texts: List[str]) -> None:
    """Insert chunks into SQLite with FTS5 index. Runs in thread."""
    def _insert():
        con = _conn()
        _init(con)

        rows = []
        for chunk, cleaned in zip(chunks, cleaned_texts):
            doc_meta = chunk.get("doc_metadata", {})
            doc_meta_payload = {
                **doc_meta,
                "bbox":           chunk.get("bbox"),
                "image_width_px": chunk.get("image_width_px"),
                "image_height_px": chunk.get("image_height_px"),
            }
            rows.append((
                chunk["chunk_id"],
                doc_meta.get("doc_id", ""),
                chunk.get("chunk_type", ""),
                chunk.get("chunk_text", ""),
                cleaned,
                int(chunk.get("page", 0)),
                chunk.get("image", ""),          # relative to settings.images_dir
                chunk.get("section_title"),
                int(chunk.get("token_count", 0)),
                chunk.get("extraction_confidence", ""),
                doc_meta.get("source_file", ""),
                json.dumps(doc_meta_payload),
            ))

        con.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, chunk_type, chunk_text, cleaned_text,
                page, image_path, section_title, token_count,
                extraction_confidence, source_file, doc_metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        con.commit()
        con.close()

    await asyncio.to_thread(_insert)
    print(f"  [text_store] inserted {len(chunks)} rows into SQLite")


def _build_fts_query(query: str, operator: str = "OR") -> Optional[str]:
    """Build FTS5 query from natural language input."""
    if any(op in query.upper() for op in (" AND ", " OR ", " NOT ", " NEAR ")):
        return query
    if any(ch in query for ch in ('"', "(", ")")):
        return query

    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return None

    if operator not in {"OR", "AND"}:
        raise ValueError("operator must be 'OR' or 'AND'")

    joiner = f" {operator} "
    return joiner.join(f'"{token}"' for token in tokens)


async def query(
    user_query: str,
    top_k: int,
    doc_id: Optional[str] = None,
    operator: str = "OR",
) -> List[Dict[str, Any]]:
    """BM25 full-text search with OR semantics by default."""
    def _query():
        con = _conn()
        fts_q = _build_fts_query(user_query, operator=operator)
        if not fts_q:
            con.close()
            return []

        doc_filter = "AND c.doc_id = ?" if doc_id else ""
        params = (fts_q, doc_id, top_k) if doc_id else (fts_q, top_k)

        rows = con.execute(f"""
            SELECT c.chunk_id, c.doc_id, c.chunk_type, c.chunk_text,
                   c.cleaned_text, c.page, c.image_path,
                   c.section_title, c.token_count, c.extraction_confidence,
                   c.source_file, c.doc_metadata, bm25(chunks_fts) AS bm25_score
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ?
            {doc_filter}
            ORDER BY bm25_score
            LIMIT ?
        """, params).fetchall()
        con.close()
        return rows

    rows = await asyncio.to_thread(_query)

    results = []
    for row in rows:
        doc_metadata: Dict[str, Any] = {}
        if row["doc_metadata"]:
            try:
                doc_metadata = json.loads(row["doc_metadata"])
            except json.JSONDecodeError:
                pass

        results.append({
            "chunk_id":              row["chunk_id"],
            "text":                  row["chunk_text"],
            "score":                 round(-row["bm25_score"], 6),
            "doc_id":                row["doc_id"],
            "page":                  row["page"],
            "chunk_type":            row["chunk_type"],
            "image_path":            row["image_path"],
            "section_title":         row["section_title"],
            "token_count":           row["token_count"],
            "extraction_confidence": row["extraction_confidence"],
            "bbox":                  doc_metadata.get("bbox"),
            "image_width_px":        doc_metadata.get("image_width_px"),
            "image_height_px":       doc_metadata.get("image_height_px"),
            "source_uri":            doc_metadata.get("source_uri"),
            "source_file":           row["source_file"],
            "total_pages":           doc_metadata.get("total_pages"),
            "pdf_type":              doc_metadata.get("pdf_type"),
            "created_at":            doc_metadata.get("created_at"),
            "doc_metadata":          doc_metadata,
            "retrieval_type":        "text",
        })
    return results
