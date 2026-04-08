"""LLM client for generating natural-language protein descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

import httpx

from graph.neo4j_client import Neo4jClient
from llm.prompts import SYSTEM_PROMPT, build_multi_protein_prompt, build_user_prompt


@dataclass
class LLMConfig:
    """Configuration for an OpenAI-compatible chat-completions endpoint."""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 700

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        Load configuration from environment variables:
        - LLM_API_KEY (required)
        - LLM_MODEL (optional)
        - LLM_BASE_URL (optional)
        - LLM_TIMEOUT_SECONDS (optional)
        - LLM_TEMPERATURE (optional)
        - LLM_MAX_TOKENS (optional)
        """
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing LLM_API_KEY environment variable.")

        return cls(
            api_key=api_key,
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "700")),
        )


class ProteinDescriptionLLMClient:
    """
    Fetch protein context from Neo4j and ask an LLM to summarize it.

    Expected graph method:
      neo4j.get_protein_context(accession) -> dict
    """

    def __init__(self, neo4j: Neo4jClient, config: LLMConfig) -> None:
        self.neo4j = neo4j
        self.config = config
        self.session = httpx.Client(
            timeout=self.config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "ProteinDescriptionLLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    _STRIP_FIELDS = frozenset({
        # bulk URL lists
        "pdb_pdb_urls", "pdb_cif_urls", "pdb_entry_urls",
        "alphafold_pdb_urls", "alphafold_cif_urls",
        "alphafold_cif_url", "alphafold_pdb_url",
        # low-value metadata
        "string_id", "ec_numbers", "is_reviewed", "taxon_id",
    })

    _PUBMED_RE = re.compile(r"\s*\(PubMed:[^)]+\)")

    def _trim_protein(self, protein: dict[str, Any]) -> dict[str, Any]:
        """Strip URL fields, low-value metadata, and PubMed citations from function_text."""
        result = {k: v for k, v in protein.items() if k not in self._STRIP_FIELDS}
        if isinstance(result.get("function_text"), str):
            result["function_text"] = self._PUBMED_RE.sub("", result["function_text"]).strip()
        return result

    def _context_for_prompt(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Compact context: protein identity + structural features only.
        URL fields, GO terms, and articles are excluded to keep prompts lean.
        """
        features = context.get("features", [])
        feature_rels = context.get("feature_relationships", [])
        protein = context.get("protein") or {}

        return {
            "protein": self._trim_protein(protein),
            "features": features,
            "feature_relationships": feature_rels,
        }

    def _build_messages(
        self,
        accession: str,
        context: dict[str, Any],
        structure_files: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        compact = self._context_for_prompt(context)
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(accession, compact, structure_files=structure_files),
            },
        ]

    def _context_for_network_prompt(
        self,
        contexts: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build a multi-protein prompt context that preserves all graph objects
        (protein, features, GO terms, relationships, and articles) while
        truncating very large article full-text fields for token safety.
        """
        return {
            acc: {
                "protein": self._trim_protein(ctx.get("protein") or {}),
                "counts": {"features": len(ctx.get("features", []))},
                "features": ctx.get("features", []),
                "feature_relationships": ctx.get("feature_relationships", []),
            }
            for acc, ctx in contexts.items()
        }

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        resp = self.session.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions", json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def describe_protein(
        self,
        accession: str,
        structure_files: list[dict[str, Any]] | None = None,
    ) -> str:
        """Fetch Neo4j context for accession and return an LLM-generated description."""
        context = self.neo4j.get_protein_context(accession)
        if not context.get("protein"):
            raise ValueError(f"Protein not found in graph: {accession}")
        messages = self._build_messages(accession, context, structure_files=structure_files)
        return self._chat(messages)

    def describe_protein_network(
        self,
        accessions: list[str],
        contexts: dict[str, dict[str, Any]],
        interaction_edges: list[dict[str, Any]],
        structure_manifest: list[dict[str, Any]],
    ) -> str:
        """Generate a concise network narrative for multiple interacting proteins."""
        packed_contexts = self._context_for_network_prompt(contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_multi_protein_prompt(
                    accessions=accessions,
                    contexts=packed_contexts,
                    interaction_edges=interaction_edges,
                    structure_manifest=structure_manifest,
                ),
            },
        ]
        return self._chat(messages)
