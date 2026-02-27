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
        "Primary objective:\n"
        "- Interleave text with structure placeholders wherever structure evidence is discussed.\n"
        "- Use placeholders exactly in the form [structure/<filename>].\n"
        "- Only use filenames that appear in the structure file manifest.\n\n"
        "Writing requirements:\n"
        "- Write coherent prose; no bullets, numbered lists, or section headers.\n"
        f"- {detail_guidance}\n"
        "- Ensure every listed protein is explicitly covered in the narrative.\n"
        "- Explain likely interaction-driven functional relationships across proteins.\n"
        "- Use interaction edge properties (for example score/confidence) when available.\n"
        "- Ground claims in the provided context and article evidence only.\n\n"
        "Coverage requirements:\n"
        "1) concise identity/function for each protein\n"
        "2) interaction network interpretation using provided INTERACTS_WITH edges\n"
        "3) structure-backed interpretation with placeholders\n"
        "4) key feature/domain and GO role connections\n"
        "5) evidence quality and caveats\n\n"
        "Important completeness rule:\n"
        "- Use the full provided graph context for each protein: protein node, all features, all GO terms,\n"
        "  all article links, and relationship properties.\n\n"
        f"Structure file manifest:\n{structure_manifest}\n\n"
        f"Interaction edges:\n{interaction_edges}\n\n"
        f"Context by accession:\n{contexts}"
    )
