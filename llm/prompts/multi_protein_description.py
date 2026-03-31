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
        f"Generate a detailed scientific report for the following interacting proteins: "
        f"{accessions}. "
        "Write entirely in flowing prose with no headers, no bullet points, and no numbered lists. "
        "The text should read as a single continuous scientific narrative.\n\n"
        f"{detail_guidance} Begin by covering each protein in turn: its full name, gene, organism, "
        "and family; its biological function and mechanism of action; GO-based biological roles "
        "from the context; key interaction partners and substrates; disease relevance if present; "
        "and how the interaction edge scores reflect the confidence of relationships between the "
        "proteins. Every listed protein must be covered.\n\n"
        "Then continue — without any section break or header — into a detailed structural analysis "
        "for each protein. Work through every domain and feature in the context: for each one, "
        "state its name, type, and residue range, then immediately place a structure placeholder "
        "[structure/<filename>] at the end of that sentence pointing to the most relevant file "
        "from the manifest (prefer a feature-slice file if one exists, otherwise a full PDB or "
        "AlphaFold file). Explain the functional significance of each region and note how shared "
        "structural motifs or interaction interfaces connect the proteins. "
        "After all features are covered, weave in any remaining manifest files not yet cited.\n\n"
        "Placeholder rules: place each placeholder inline at the sentence end, not on its own line; "
        "only use filenames verbatim from the manifest; every manifest file must appear at least once.\n\n"
        f"Structure file manifest:\n{structure_manifest}\n\n"
        f"Interaction edges:\n{interaction_edges}\n\n"
        f"Context by accession:\n{contexts}"
    )
