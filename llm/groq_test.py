"""Quick model comparison: run single-protein describe on two Groq models."""
import os, json, textwrap
import httpx
from graph.neo4j_client import Neo4jClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password123")
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
BASE_URL   = "https://api.groq.com/openai/v1/chat/completions"
ACCESSION  = "P31749"  # AKT1

MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

def compact_context(ctx: dict) -> dict:
    features = ctx.get("features", [])
    feature_rels = ctx.get("feature_relationships", [])
    return {
        "protein": ctx.get("protein"),
        "counts": {"features": len(features)},
        "features": features,
        "feature_relationships": feature_rels,
    }

def call_groq(model: str, messages: list[dict]) -> str:
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1200}
    r = httpx.post(BASE_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def main():
    with Neo4jClient(uri=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS) as neo4j:
        ctx = neo4j.get_protein_context(ACCESSION)

    compact = compact_context(ctx)
    prompt_chars = len(json.dumps(compact))
    print(f"Protein: {ACCESSION} | Context: {prompt_chars} chars (~{prompt_chars//4} tokens)\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(ACCESSION, compact, structure_files=[])},
    ]

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print('='*70)
        try:
            text = call_groq(model, messages)
            print(text)
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
