"""Physiological meal perturbations for paired policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import jax.numpy as jnp
import numpy as np


@dataclass
class HiddenMealCondition:
    """Leave controller observations intact and record a physiological meal."""

    minute: int
    grams: float
    run_tag: str = ""

    def __post_init__(self) -> None:
        if self.minute < 0 or self.minute % 5:
            raise ValueError("Meal minute must be a nonnegative five-minute boundary.")
        if not np.isfinite(self.grams) or self.grams < 0:
            raise ValueError("Meal grams must be finite and nonnegative.")

    @property
    def output_tag(self) -> str:
        tag = f"hidden_meal_m{self.minute}_g{self.grams:g}".replace(".", "p")
        return f"{tag}_{self.run_tag}" if self.run_tag else tag

    def reset(self, *, episode_index: int, seed: int) -> None:
        del episode_index, seed

    def apply(
        self,
        observation: np.ndarray,
        *,
        episode_index: int,
        step_index: int,
    ) -> np.ndarray:
        del episode_index, step_index
        return np.array(observation, dtype=np.float32, copy=True)

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.output_tag,
            "output_tag": self.output_tag,
            "observation": [],
            "physiology": {
                "type": "hidden_meal",
                "parameters": {"minute": self.minute, "grams": self.grams},
                "frequency": "once_per_episode",
                "announced_in_meal_preview": False,
            },
        }


class HiddenMealEnv:
    """Attach an unannounced meal to the group's single-patient simulator."""

    def __init__(self, env: Any, *, minute: int, grams: float) -> None:
        self._wrapped = env
        self.minute = minute
        self.grams = grams
        self.traces: List[Dict[str, Any]] = []
        self._trace: Dict[str, Any] = {}
        self._observation: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def _simulator(self) -> Any:
        patients = self._wrapped.env.envs
        if len(patients) != 1:
            raise ValueError("Hidden-meal evaluation requires one patient environment.")
        return patients[0].unwrapped

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observation, info = self._wrapped.reset(*args, **kwargs)
        simulator = self._simulator()
        horizon = int(simulator.env_params.simulation_minutes)
        if not 0 <= self.minute < horizon:
            raise ValueError("Meal minute must fall within the simulation horizon.")
        if int(np.asarray(simulator._jax_state["t"])) != 0:
            raise ValueError("Expected an episode starting at simulator minute zero.")

        schedule = jnp.zeros(horizon, dtype=jnp.float32)
        schedule = schedule.at[self.minute].set(self.grams)
        # Inject after reset so warmup and the announced meal preview stay intact.
        # Clean runs use a zero schedule to keep the paired JAX state structure equal.
        simulator._jax_state = {**simulator._jax_state, "hidden_meals": schedule}
        self._trace = {
            "minutes": [],
            "observations": [],
            "actions": [],
            "meal_delivered": False,
        }
        self.traces.append(self._trace)
        self._observation = observation
        return observation, info

    def step(self, action: Any) -> Any:
        simulator = self._simulator()
        minute = int(np.asarray(simulator._jax_state["t"]))
        self._trace["minutes"].append(minute)
        self._trace["observations"].append(np.asarray(self._observation).copy())
        self._trace["actions"].append(np.asarray(action).copy())
        result = self._wrapped.step(action)
        state = simulator._jax_state
        if "hidden_meals" not in state:
            raise RuntimeError(
                "Hidden-meal state was lost. Apply patches/hidden_meal.patch first."
            )
        next_minute = int(np.asarray(state["t"]))
        if minute <= self.minute < next_minute and self.grams > 0:
            self._trace["meal_delivered"] = True
        self._observation = result[0]
        return result


def verify_hidden_meal_pair(
    clean: List[Dict[str, Any]],
    meal: List[Dict[str, Any]],
    *,
    minute: int,
) -> None:
    """Require identical controller inputs and actions before meal ingestion."""
    if len(clean) != len(meal):
        raise RuntimeError("Paired runs have different episode counts.")
    for episode, (left, right) in enumerate(zip(clean, meal)):
        if not right["meal_delivered"]:
            raise RuntimeError(f"Episode {episode} ended before the hidden meal.")
        for field in ("minutes", "observations", "actions"):
            a = np.asarray(left[field])[np.asarray(left["minutes"]) <= minute]
            b = np.asarray(right[field])[np.asarray(right["minutes"]) <= minute]
            if not np.array_equal(a, b):
                raise RuntimeError(f"Episode {episode}: pre-meal {field} differ.")
