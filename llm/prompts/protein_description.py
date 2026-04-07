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
        f"Generate a natural-language description for protein accession {accession}.\n\n"
        "Write flowing scientific prose with no headers, bullets, or numbered lists. "
        "Cover the protein's identity and function, its key structural features and domains, "
        "and any available structural evidence from PDB or AlphaFold entries. "
        "When referencing a structure, insert its placeholder exactly as it appears "
        "in the 'placeholder' field of the structure manifest below, inline at the "
        "relevant point in the text. Only use placeholders from the manifest. "
        "Do not invent or speculate beyond what the context provides.\n\n"
        f"Structure manifest:\n{structure_files}\n\n"
        f"Context JSON:\n{context}"
    )


