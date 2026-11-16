"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol
import requests


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """TODO: implement a provider adapter; do not silently fall back to mock vectors."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.dimensions = 768 if "gemma" in model_name.lower() else 384
        self.api_url = "http://localhost:11434/api/embed"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = requests.post(
            self.api_url,
            json={"model": self.model_name, "input": [f"title: none | text: {t}" for t in texts]}
        )
        response.raise_for_status()
        return response.json().get("embeddings", [])

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            return []
        response = requests.post(
            self.api_url,
            json={"model": self.model_name, "input": f"task: search result | query: {question}"}
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        return embeddings[0] if embeddings else []