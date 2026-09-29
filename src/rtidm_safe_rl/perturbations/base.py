"""Interfaces shared by observation perturbations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class PerturbationContext:
    """Location of a controller decision within an evaluation run."""

    episode_index: int
    step_index: int
    episode_seed: int


class ObservationPerturbation(ABC):
    """Transform a controller-only copy of an environment observation."""

    type_name = "observation_perturbation"

    def reset(self, *, episode_index: int, seed: int) -> None:
        """Reset per-episode state; stateful perturbations may override this."""
        del episode_index, seed

    @abstractmethod
    def apply(
        self,
        observation: np.ndarray,
        *,
        context: PerturbationContext,
    ) -> np.ndarray:
        """Return the observation visible to the controller."""

    def metadata(self) -> Dict[str, Any]:
        """Return serializable configuration metadata."""
        return {"type": self.type_name}
