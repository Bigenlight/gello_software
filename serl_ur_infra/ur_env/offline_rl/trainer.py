"""One offline training step; no robot, replay ingress or online update budget."""

from dataclasses import dataclass, replace

from .interfaces import Algorithm, ChunkDataset


@dataclass(frozen=True)
class OfflineTrainer:
    algorithm: Algorithm
    batch_size: int = 256
    horizon: int = 16
    step: int = 0

    def __post_init__(self):
        for name in ("batch_size", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def train_once(self, dataset: ChunkDataset, *, rng):
        batch = dataset.sample_chunks(self.batch_size, self.horizon)
        batch.validate()
        if batch.actions.shape[:2] != (self.batch_size, self.horizon):
            raise ValueError("dataset returned a different batch size or horizon")
        algorithm, metrics = self.algorithm.update(batch, rng=rng)
        return replace(self, algorithm=algorithm, step=self.step + 1), metrics
