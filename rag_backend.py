from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
# Environment-driven configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# Deployments
CHAT_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Lazy-initialized client
_client: Optional[AzureOpenAI] = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
            raise RuntimeError(
                "Missing Azure OpenAI configuration. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY.")
        _client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=API_VERSION,
        )
    return _client


def rag_reply(query: str, hf_token: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, List[Dict[str, str]]]:
    """
    Chat backend using Azure OpenAI Chat Completions.
    - query: user message
    - hf_token: ignored (for compatibility with previous interface)
    - history: list of {role: "user"|"assistant"|"system", content: str}
    Returns (assistant_text, updated_history)
    """
    client = _get_client()

    messages: List[Dict[str, str]] = []
    # System prompt to make the bot movie-intelligent
    system_prompt = (
        "You are MovieMate, a helpful, concise movie expert. "
        "Answer about films, casts, directors, genres, and recommendations. "
        "When suggesting movies, include brief reasons. If unsure, say so clearly."
        "Just answer movie related questions and actors, disregard others."
    )
    messages.append({"role": "system", "content": system_prompt})

    if history:
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.4,
        max_tokens=400,
    )

    text = resp.choices[0].message.content if resp.choices else ""
    updated = (history or []) + [{"role": "assistant", "content": text}]
    return text, updated
