"""Prompt templates for protein-description generation."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a biomedical assistant. Write a concise, factual description of a protein "
    "using the provided graph-derived context. If evidence is limited, explicitly say so. "
    "Do not invent facts."
)


def build_user_prompt(accession: str, context: dict[str, Any]) -> str:
    """Build the user prompt for protein-description generation."""
    return (
        f"Generate a natural-language summary for protein accession {accession}.\n\n"
        "Please cover:\n"
        "1) identity/function\n"
        "2) structural evidence (PDB/AlphaFold)\n"
        "3) notable feature/domain annotations\n"
        "4) GO-based biological role\n"
        "5) publication evidence quality/caveats\n\n"
        f"Context JSON:\n{context}"
    )
