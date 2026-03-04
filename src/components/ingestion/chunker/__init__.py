"""
chunker — PDF chunking library.

Public API:
    from src.components.ingestion.chunker import PDFChunker
    from src.config import ChunkingConfig
"""

from src.components.ingestion.chunker.chunker import PDFChunker

__all__ = ["PDFChunker"]
__version__ = "6.0.0"
