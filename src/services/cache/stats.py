from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStatsSnapshot:
    hits: int
    misses: int
    bypasses: int
    writes: int
    read_failures: int
    write_failures: int
    invalid_entries: int

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class CacheStats:
    """Process-local best-effort accounting for the shared RAG response cache."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.writes = 0
        self.read_failures = 0
        self.write_failures = 0
        self.invalid_entries = 0

    def increment(self, name: str) -> None:
        try:
            setattr(self, name, getattr(self, name) + 1)
        except Exception:
            # Accounting must never affect RAG correctness.
            return

    def snapshot(self) -> CacheStatsSnapshot:
        return CacheStatsSnapshot(
            hits=self.hits,
            misses=self.misses,
            bypasses=self.bypasses,
            writes=self.writes,
            read_failures=self.read_failures,
            write_failures=self.write_failures,
            invalid_entries=self.invalid_entries,
        )
