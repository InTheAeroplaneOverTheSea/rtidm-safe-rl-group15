"""Built-in observation perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ObservationPerturbation, PerturbationContext


@dataclass
class PersistentCGMBias(ObservationPerturbation):
    """Add a fixed mg/dL offset to CGM feature 0 for the full episode."""

    bias_mgdl: float
    type_name = "persistent_cgm_bias"

    def __post_init__(self) -> None:
        self.bias_mgdl = float(self.bias_mgdl)
        if not np.isfinite(self.bias_mgdl):
            raise ValueError("bias_mgdl must be finite.")

    def apply(
        self,
        observation: np.ndarray,
        *,
        context: PerturbationContext,
    ) -> np.ndarray:
        del context
        corrupted = np.array(observation, dtype=np.float32, copy=True)
        if corrupted.shape[-1] < 1:
            raise ValueError("Observation must contain CGM at feature index 0.")
        corrupted[..., 0] += self.bias_mgdl
        return corrupted

    def metadata(self):
        return {"type": self.type_name, "parameters": {"bias_mgdl": self.bias_mgdl}}
