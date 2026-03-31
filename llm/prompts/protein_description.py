"""Prompt templates for protein-description generation."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a structural bioinformatics expert. Given structured data about a protein, "
    "produce a detailed scientific description focused on its identity, function, and "
    "structural features. Be precise and factual. Do not invent information not present "
    "in the provided context."
)


def build_user_prompt(
    accession: str,
    context: dict[str, Any],
    structure_files: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user prompt for protein description and structural analysis."""
    structure_files = structure_files or []
    return (
        f"Generate a detailed scientific report for protein accession {accession}. "
        "Write entirely in flowing prose with no headers, no bullet points, and no numbered lists. "
        "The text should read as a single continuous scientific narrative.\n\n"
        "Begin with several paragraphs covering the protein's full name, gene, organism, and family; "
        "its biological function and mechanism of action; GO-based biological roles from the context; "
        "key interaction partners and substrates; and any disease relevance present in the context.\n\n"
        "Then continue — without any section break or header — into a detailed structural analysis. "
        "Work through every domain and feature in the context: for each one, state its name, type, "
        "and residue range, then immediately place a structure placeholder [structure/<filename>] "
        "at the end of that sentence pointing to the most relevant file from the manifest "
        "(prefer a feature-slice file for that domain if one exists, otherwise use a full PDB or "
        "AlphaFold file). Explain the functional significance of each region. "
        "After all features are covered, weave in any remaining manifest files not yet cited.\n\n"
        "Placeholder rules: place each placeholder inline at the sentence end, not on its own line; "
        "only use filenames that appear verbatim in the manifest; every manifest file must appear at least once.\n\n"
        f"Structure file manifest:\n{structure_files}\n\n"
        f"Context JSON:\n{context}"
    )


