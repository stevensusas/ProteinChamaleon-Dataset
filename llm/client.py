"""LLM client for generating natural-language protein descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import os
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

    def _context_for_prompt(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Keep prompt compact by sampling large collections.
        The full graph context can be very large for proteins like TP53.
        """
        features = context.get("features", [])
        feature_rels = context.get("feature_relationships", [])
        go_terms = context.get("go_terms", [])
        go_rels = context.get("go_relationships", [])
        articles = context.get("articles", [])
        article_rels = context.get("article_relationships", {})

        return {
            "protein": context.get("protein"),
            "counts": {
                "features": len(features),
                "go_terms": len(go_terms),
                "articles": len(articles),
                "cited_in_edges": len(article_rels.get("cited_in", [])),
                "feature_evidenced_by_edges": len(article_rels.get("feature_evidenced_by", [])),
            },
            "features_sample": features[:20],
            "feature_relationships_sample": feature_rels[:30],
            "go_terms_sample": go_terms[:30],
            "go_relationships_sample": go_rels[:40],
            "articles_sample": articles[:25],
            "article_relationships_sample": {
                "cited_in": article_rels.get("cited_in", [])[:40],
                "feature_evidenced_by": article_rels.get("feature_evidenced_by", [])[:40],
            },
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
        packed: dict[str, Any] = {}
        for acc, ctx in contexts.items():
            articles = ctx.get("articles", []) or []
            packed_articles: list[dict[str, Any]] = []
            for article in articles:
                if not isinstance(article, dict):
                    continue
                art = dict(article)
                full_text = art.get("full_text")
                if isinstance(full_text, str):
                    art["full_text_excerpt"] = full_text[:1200]
                    art["full_text_char_count"] = len(full_text)
                    art.pop("full_text", None)
                packed_articles.append(art)

            packed[acc] = {
                "protein": ctx.get("protein"),
                "features": ctx.get("features", []),
                "feature_relationships": ctx.get("feature_relationships", []),
                "go_terms": ctx.get("go_terms", []),
                "go_relationships": ctx.get("go_relationships", []),
                "articles": packed_articles,
                "article_relationships": ctx.get("article_relationships", {}),
            }
        return packed

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
