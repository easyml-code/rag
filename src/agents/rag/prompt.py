from __future__ import annotations

from typing import Any, Dict


SYSTEM_PROMPT = """You are a retrieval-grounded assistant.

Rules you must follow:
1) Use only the retrieved sources provided in context.
2) Every factual claim must have inline citation(s) in this exact format: [n]
3) Citations must be attached to the previous token with no space, e.g. 1001 MW[1].
4) Citation n must map to one of the provided source numbers.
5) Do not invent citations or sources.
6) If relevant evidence is not present, reply exactly: Not found in the document.
7) Keep responses concise and directly answer the user.
"""


def build_context_block(
    sources_by_citation: Dict[int, Dict[str, Any]],
    include_text: bool = True,
) -> str:
    """
    Build the LLM context block for retrieved sources.

    include_text=True  - full chunk text included (images=False OR images_with_text=True)
    include_text=False - metadata anchor only; images carry the evidence (images_with_text=False)
    """
    if not sources_by_citation:
        return "No sources were retrieved."

    blocks = []
    for n, source in sources_by_citation.items():
        lines = [
            f"[{n}]",
            f"chunk_id: {source.get('chunk_id', '')}",
            f"page: {source.get('page', '')}",
            f"chunk_type: {source.get('chunk_type', '')}",
            f"source_file: {source.get('source_file', '')}",
        ]
        if include_text:
            text = str(source.get("text", "")).strip()[:2000]
            lines.append(f"text: {text}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)