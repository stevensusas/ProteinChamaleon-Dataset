"""Prompt templates for protein-description generation."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a biomedical assistant. Write a concise, factual description of a protein "
    "using the provided graph-derived context. If evidence is limited, explicitly say so. "
    "Do not invent facts. Use coherent scientific prose, not bullet points."
)


def build_user_prompt(
    accession: str,
    context: dict[str, Any],
    structure_files: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user prompt for protein-description generation."""
    structure_files = structure_files or []
    return (
        f"Generate a natural-language summary for protein accession {accession}.\n\n"
        "Output format requirements:\n"
        "- Write as a coherent scientific narrative in prose.\n"
        "- Do not use bullets, numbered lists, or section headers.\n"
        "- Keep the flow logical across function, structure, features, GO role, and evidence.\n\n"
        "Structure placeholder requirements:\n"
        "- Interleave text with structure placeholders when structure evidence is discussed.\n"
        "- Use placeholders exactly in the form [structure/<filename>].\n"
        "- Only use filenames from the provided structure file manifest.\n\n"
        "Please cover:\n"
        "1) identity/function\n"
        "2) structural evidence (PDB/AlphaFold)\n"
        "3) notable feature/domain annotations\n"
        "4) GO-based biological role\n"
        "5) publication evidence quality/caveats\n\n"
        f"Structure file manifest:\n{structure_files}\n\n"
        f"Context JSON:\n{context}"
    )


