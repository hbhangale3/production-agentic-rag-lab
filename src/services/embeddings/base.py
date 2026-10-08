from typing import Protocol


class EmbeddingProvider(Protocol):
    """Provider-independent text embedding boundary."""

    def initialize(self) -> None:
        """Load provider resources once when explicit warm-up is desired."""
        ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed retrieval passages in input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""
        ...
