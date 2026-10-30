"""Small extension points; each algorithm owns its update order."""

from typing import Any, Mapping, Protocol

from .batch import ChunkBatch

Metrics = Mapping[str, Any]
# Optional algorithm-owned cache: e.g. endcorr candidates and flow trajectories.
UpdateContext = Any


class Actor(Protocol):
    def update(self, batch: ChunkBatch, *, critic: "Critic",
               context: UpdateContext, rng: Any) -> tuple["Actor", Metrics]: ...

    def sample_candidates(self, observations: Any, *, rng: Any, count: int) -> Any:
        """Return (B,N,H,A) candidates in the same action contract as training."""
        ...


class Critic(Protocol):
    def update(self, batch: ChunkBatch, *, actor: Actor,
               context: UpdateContext, rng: Any) -> tuple["Critic", Metrics, UpdateContext]: ...

    def score(self, observations: Any, actions: Any) -> Any:
        """Return (B,) scores for (B,H,A) actions."""
        ...


class Algorithm(Protocol):
    def update(self, batch: ChunkBatch, *, rng: Any) -> tuple["Algorithm", Metrics]: ...


class ChunkDataset(Protocol):
    def sample_chunks(self, batch_size: int, horizon: int) -> ChunkBatch:
        """Sample fixed data; own RNG, episode boundaries and reward conversion."""
        ...
