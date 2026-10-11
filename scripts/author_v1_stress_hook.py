#!/usr/bin/env python3
"""Stress and deterministic episode-seed adapter for author-v1 ``eval_run``.

This module is copied beside the immutable January GlucoAlg evaluator by the
SURF job.  It deliberately changes only evaluation setup: policy checkpoints,
the BA-NODE predictor, GlucoAlg, and GlucoSim remain pinned to author-v1.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

from interface_new import GlucoSimInterface, Meal, StressConfig


STRESS_SCENARIOS = {
    "nominal": StressConfig(),
    "sensor_sigma15": StressConfig(cgm_noise_sigma=15.0),
    # One surprise meal every simulated day.  It is injected only after the
    # policy and Shield have selected their action for 15:00.
    "unannounced_meal_60g": StressConfig(
        unannounced_daily_meals=[Meal("15:00", 60.0)],
    ),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _episode_seed(base_seed: int, replicate_id: int, episode: int) -> int:
    """Condition-independent uint32 seed for an exactly paired episode."""
    sequence = np.random.SeedSequence(
        [int(base_seed), int(replicate_id), int(episode), 20261011]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class AuthorV1StressController:
    """Bridge Sherlock's raw Gym interface to author-v1 ``DiabetesEnvs``."""

    def __init__(
        self,
        wrapped_env,
        *,
        scenario: str,
        base_seed: int,
        replicate_id: int,
    ) -> None:
        if scenario not in STRESS_SCENARIOS:
            choices = ", ".join(sorted(STRESS_SCENARIOS))
            raise ValueError(f"Unknown stress scenario {scenario!r}; choose {choices}")
        if base_seed < 0 or replicate_id < 0:
            raise ValueError("base_seed and replicate_id must be nonnegative")

        vector_env = getattr(wrapped_env, "env", None)
        envs = getattr(vector_env, "envs", None)
        if not envs or len(envs) != 1:
            raise ValueError("Author-v1 stress evaluation requires exactly one raw env")

        self.raw_env = envs[0]
        self.scenario = scenario
        self.stress = STRESS_SCENARIOS[scenario]
        self.base_seed = int(base_seed)
        self.replicate_id = int(replicate_id)
        self.interface = GlucoSimInterface(
            patient_name=str(self.raw_env.env_params.patient_name),
            simulation_minutes=int(wrapped_env.simulation_minutes),
            sample_time=int(wrapped_env.sample_time),
        )
        # Sensor configuration participates in GlucoSim's warm-state cache key
        # and must therefore be installed before the first reset.
        self.interface.configure_existing_env(self.raw_env, self.stress)
        self.episode_records: list[dict[str, Any]] = []
        self._current_record: dict[str, Any] | None = None

    def reset_episode(self, wrapped_env, episode: int):
        """Reset with a key independent of prior episode length or condition."""
        seed = _episode_seed(self.base_seed, self.replicate_id, episode)
        # January GlucoSim ignores reset(seed=...) once ``key`` exists.  Setting
        # the raw key immediately before reset makes every episode independent
        # of how many steps the previous policy happened to survive.
        self.raw_env.key = jax.random.PRNGKey(seed)
        obs, info = wrapped_env.reset(seed=seed)
        corrected = self.interface.after_reset(self.raw_env, self.stress, seed)
        if corrected is not None:
            obs = torch.as_tensor(
                np.asarray(corrected),
                dtype=torch.float32,
                device=getattr(wrapped_env, "_device", torch.device("cpu")),
            ).unsqueeze(0)

        schedule = np.asarray(
            self.raw_env._jax_state["scenario_meals"], dtype=np.float32
        )
        noise = self.raw_env.env_params.noise_config
        record = {
            "episode": int(episode),
            "episode_seed": seed,
            "meal_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "planned_unannounced_meals": (
                self.interface.planned_unannounced_meal_events(self.stress)
            ),
            "delivered_unannounced_meals": [],
            "cgm_obs_sigma_mgdl": float(noise.cgm_obs_sigma_mgdl),
        }
        self.episode_records.append(record)
        self._current_record = record
        return obs, info

    def before_step(self) -> list[dict]:
        """Inject surprise carbohydrates after Shield action selection."""
        due = self.interface.inject_unannounced_meals(self.raw_env, self.stress)
        if due:
            if self._current_record is None:
                raise RuntimeError("before_step called before reset_episode")
            self._current_record["delivered_unannounced_meals"].extend(due)
        return due

    def write_metadata(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / "stress_episode_metadata.json"
        payload = {
            "schema_version": 1,
            "stress_scenario": self.scenario,
            "stress_config": _jsonable(asdict(self.stress)),
            "base_evaluation_seed": self.base_seed,
            "replicate_id": self.replicate_id,
            "episode_seed_policy": (
                "numpy.SeedSequence([base_seed, replicate_id, episode, 20261011])"
            ),
            "unannounced_meal_semantics": (
                "inject pending carbohydrates after policy/Shield action selection "
                "and before env.step; never add the event to scenario_meals"
            ),
            "episodes": self.episode_records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path
