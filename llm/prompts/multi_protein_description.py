"""Prompt template for multi-protein network description generation."""

from __future__ import annotations

from typing import Any


def build_multi_protein_prompt(
    accessions: list[str],
    contexts: dict[str, dict[str, Any]],
    interaction_edges: list[dict[str, Any]],
    structure_manifest: list[dict[str, Any]],
) -> str:
    """Build prompt for concise, interleaved multi-protein network descriptions."""
    protein_count = max(1, len(accessions))
    detail_guidance = (
        "Provide moderate detail per protein."
        if protein_count <= 2
        else "Be concise per protein and prioritize cross-protein interaction insights."
    )
    return (
        f"Generate a network-focused scientific narrative for proteins: {accessions}.\n\n"
        "Write flowing scientific prose with no headers, bullets, or numbered lists. "
        f"{detail_guidance} "
        "Every listed protein must be explicitly covered in the narrative. "
        "Describe each protein's identity and key structural features and domains, "
        "then interpret the interaction network using the provided INTERACTS_WITH edges, "
        "including any available score or confidence properties. "
        "When referencing a structure, insert its placeholder exactly as it appears "
        "in the 'placeholder' field of the structure manifest below, inline at the "
        "relevant point in the text. Only use placeholders from the manifest. "
        "Do not invent or speculate beyond what the context provides.\n\n"
        f"Structure manifest:\n{structure_manifest}\n\n"
        f"Interaction edges:\n{interaction_edges}\n\n"
        f"Context by accession:\n{contexts}"
    )
