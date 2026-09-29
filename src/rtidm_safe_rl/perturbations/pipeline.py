"""Composition and lifecycle management for observation perturbations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .base import ObservationPerturbation, PerturbationContext


def _safe_output_tag(name: str) -> str:
    tag = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not tag:
        raise ValueError("Perturbation name must contain at least one letter or digit.")
    return tag


@dataclass
class ObservationPerturbationPipeline:
    """Apply configured perturbations in order to a controller-only copy."""

    name: str
    perturbations: List[ObservationPerturbation] = field(default_factory=list)
    _episode_index: int = field(default=0, init=False, repr=False)
    _episode_seed: int = field(default=0, init=False, repr=False)

    @property
    def output_tag(self) -> str:
        return _safe_output_tag(self.name) if self.perturbations else ""

    def reset(self, *, episode_index: int, seed: int) -> None:
        self._episode_index = int(episode_index)
        self._episode_seed = int(seed)
        child_seeds = np.random.SeedSequence(seed).spawn(len(self.perturbations))
        for perturbation, child_seed in zip(self.perturbations, child_seeds):
            derived_seed = int(child_seed.generate_state(1, dtype=np.uint32)[0])
            perturbation.reset(episode_index=episode_index, seed=derived_seed)

    def apply(
        self,
        observation: np.ndarray,
        *,
        episode_index: int,
        step_index: int,
    ) -> np.ndarray:
        context = PerturbationContext(
            episode_index=int(episode_index),
            step_index=int(step_index),
            episode_seed=self._episode_seed,
        )
        transformed = np.array(observation, dtype=np.float32, copy=True)
        for perturbation in self.perturbations:
            transformed = perturbation.apply(transformed, context=context)
        return np.array(transformed, dtype=np.float32, copy=True)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "output_tag": self.output_tag,
            "observation": [item.metadata() for item in self.perturbations],
        }
