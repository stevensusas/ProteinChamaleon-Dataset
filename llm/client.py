"""LLM client for generating natural-language protein descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import httpx

from graph.neo4j_client import Neo4jClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt


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

    def _build_messages(self, accession: str, context: dict[str, Any]) -> list[dict[str, str]]:
        compact = self._context_for_prompt(context)
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(accession, compact),
            },
        ]

    def describe_protein(self, accession: str) -> str:
        """Fetch Neo4j context for accession and return an LLM-generated description."""
        context = self.neo4j.get_protein_context(accession)
        if not context.get("protein"):
            raise ValueError(f"Protein not found in graph: {accession}")

        payload = {
            "model": self.config.model,
            "messages": self._build_messages(accession, context),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        resp = self.session.post(f"{self.config.base_url.rstrip('/')}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
