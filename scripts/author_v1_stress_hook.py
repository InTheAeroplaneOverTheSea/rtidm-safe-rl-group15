#!/usr/bin/env python3
"""Stress and deterministic episode-seed adapter for author-v1 ``eval_run``.

This module is copied beside the immutable January GlucoAlg evaluator by the
SURF job. It changes evaluation setup and observation corruption only: policy
checkpoints, the BA-NODE predictor, GlucoAlg, and GlucoSim remain pinned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from interface_new import GlucoSimInterface, Meal, StressConfig


SCENARIO_SCHEMA_VERSION = 2
REPLICATE_SEED_STRIDE = 10_000
MARKOV_STREAM_TAG = 0x43474D  # ASCII-ish "CGM", independent of GlucoSim RNG.

CANONICAL_STRESS_SCENARIOS = {
    "nominal": StressConfig(),
    "gaussian_sigma15": StressConfig(cgm_noise_sigma=15.0),
    "bias_positive": StressConfig(cgm_bias_mgdl=10.0),
    "bias_negative": StressConfig(cgm_bias_mgdl=-10.0),
    # The 0.003 value is provisional until the CPO calibration is accepted.
    "random_walk_drift": StressConfig(cgm_rw_sigma_bias=0.003),
    "dropout_markov_v1": StressConfig(
        cgm_gap_entry_prob=0.003,
        cgm_gap_recovery_prob=0.2917,
        cgm_gap_sampling_interval_min=5,
    ),
    "unannounced_meal_60g": StressConfig(
        unannounced_daily_meals=[Meal("15:00", 60.0)],
    ),
    # Explicit pilot-only drift bracket. These are never aliases for the
    # eventual frozen primary condition.
    "drift_sigma0025_calibration": StressConfig(cgm_rw_sigma_bias=0.0025),
    "drift_sigma004_calibration": StressConfig(cgm_rw_sigma_bias=0.004),
}

SCENARIO_ALIASES = {
    "sensor_sigma15": "gaussian_sigma15",
}

# The patched author evaluator derives argparse choices from this dictionary.
# Alias values intentionally reference the canonical configuration.
STRESS_SCENARIOS = {
    **CANONICAL_STRESS_SCENARIOS,
    **{
        alias: CANONICAL_STRESS_SCENARIOS[canonical]
        for alias, canonical in SCENARIO_ALIASES.items()
    },
}

CALIBRATION_SCENARIOS = (
    "nominal",
    "gaussian_sigma15",
    "bias_positive",
    "bias_negative",
    "drift_sigma0025_calibration",
    "random_walk_drift",
    "drift_sigma004_calibration",
    "dropout_markov_v1",
)


def canonical_scenario_name(scenario: str) -> str:
    """Return the one output identifier used for a scenario and its aliases."""
    return SCENARIO_ALIASES.get(scenario, scenario)


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
    """Return literal 22..31 for base=22, replicate=0, episodes 0..9."""
    if min(base_seed, replicate_id, episode) < 0:
        raise ValueError("base_seed, replicate_id, and episode must be nonnegative")
    if episode >= REPLICATE_SEED_STRIDE:
        raise ValueError(
            f"episode must be smaller than {REPLICATE_SEED_STRIDE} to keep "
            "replicate seed blocks disjoint"
        )
    seed = int(base_seed) + int(replicate_id) * REPLICATE_SEED_STRIDE + int(episode)
    if seed > np.iinfo(np.uint32).max:
        raise ValueError(f"episode seed exceeds uint32 range: {seed}")
    return seed


def _observation_value(observation, index: int) -> float:
    array = (
        observation.detach().cpu().numpy()
        if isinstance(observation, torch.Tensor)
        else np.asarray(observation)
    )
    row = array[0] if array.ndim > 1 else array
    return float(row[index])


def _scalar(value) -> float:
    """Convert scalar or single-vector wrapper output to a Python float."""
    array = np.asarray(value).squeeze()
    if array.shape != ():
        raise ValueError(f"expected one scalar value, got shape {array.shape}")
    return float(array)


def _replace_observation_sensor(observation, cgm: float, trend: float):
    if isinstance(observation, torch.Tensor):
        updated = observation.clone()
        if updated.ndim > 1:
            updated[..., 0] = cgm
            updated[..., 3] = trend
        else:
            updated[0] = cgm
            updated[3] = trend
        return updated
    updated = np.array(observation, copy=True)
    if updated.ndim > 1:
        updated[..., 0] = cgm
        updated[..., 3] = trend
    else:
        updated[0] = cgm
        updated[3] = trend
    return updated


def _gap_intervals(schedule: np.ndarray, sample_time: int) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start: int | None = None
    for step, unavailable in enumerate(schedule.tolist() + [False]):
        if unavailable and start is None:
            start = step
        elif not unavailable and start is not None:
            intervals.append(
                {
                    "start_step": start,
                    "end_step_exclusive": step,
                    "start_time_minutes": (start + 1) * sample_time,
                    "duration_minutes": (step - start) * sample_time,
                }
            )
            start = None
    return intervals


class AuthorV1StressController:
    """Bridge the raw Gym simulator to author-v1 ``DiabetesEnvs``."""

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
        self.requested_scenario = scenario
        self.scenario = canonical_scenario_name(scenario)
        self.stress = CANONICAL_STRESS_SCENARIOS[self.scenario]
        self.base_seed = int(base_seed)
        self.replicate_id = int(replicate_id)
        self.interface = GlucoSimInterface(
            patient_name=str(self.raw_env.env_params.patient_name),
            simulation_minutes=int(wrapped_env.simulation_minutes),
            sample_time=int(wrapped_env.sample_time),
        )
        self.sample_time = int(wrapped_env.sample_time)
        self.max_controller_steps = int(wrapped_env.simulation_minutes) // self.sample_time
        # Sensor configuration participates in GlucoSim's warm-state cache key
        # and must therefore be installed before the first reset.
        self.interface.configure_existing_env(self.raw_env, self.stress)
        self.episode_records: list[dict[str, Any]] = []
        self._current_record: dict[str, Any] | None = None
        self._gap_schedule = np.zeros(self.max_controller_steps, dtype=np.bool_)
        self._controller_step = 0
        self._last_exposed_cgm: float | None = None

    def _build_gap_schedule(self, episode_seed: int) -> np.ndarray:
        entry = self.stress.cgm_gap_entry_prob
        recovery = self.stress.cgm_gap_recovery_prob
        if entry is None:
            return np.zeros(self.max_controller_steps, dtype=np.bool_)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(episode_seed), MARKOV_STREAM_TAG])
        )
        schedule = np.zeros(self.max_controller_steps, dtype=np.bool_)
        unavailable = False
        for step in range(self.max_controller_steps):
            draw = float(rng.random())
            if unavailable:
                unavailable = not (draw < float(recovery))
            else:
                unavailable = draw < float(entry)
            schedule[step] = unavailable
        return schedule

    def reset_episode(self, wrapped_env, episode: int):
        """Reset with a literal, condition-independent episode seed."""
        seed = _episode_seed(self.base_seed, self.replicate_id, episode)
        # January GlucoSim ignores reset(seed=...) once ``key`` exists. Setting
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

        self._gap_schedule = self._build_gap_schedule(seed)
        self._controller_step = 0
        self._last_exposed_cgm = _observation_value(obs, 0)

        schedule = np.asarray(
            self.raw_env._jax_state["scenario_meals"], dtype=np.float32
        )
        noise = self.raw_env.env_params.noise_config
        initial_scale = float(np.asarray(self.raw_env._jax_state["cgm_scale"]))
        record = {
            "episode": int(episode),
            "episode_seed": seed,
            "meal_schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "planned_unannounced_meals": (
                self.interface.planned_unannounced_meal_events(self.stress)
            ),
            "delivered_unannounced_meals": [],
            "planned_markov_gap_intervals": _gap_intervals(
                self._gap_schedule, self.sample_time
            ),
            "realized_markov_gap_steps": [],
            "native_dropout_minutes": 0,
            "sensor_clip_minutes": 0,
            "exposed_clip_observations": 0,
            "cgm_scale_min": initial_scale,
            "cgm_scale_max": initial_scale,
            "cgm_scale_boundary_observations": 0,
            "cgm_scale_boundary_minutes": 0,
            "noise_config": {
                "cgm_obs_sigma_mgdl": float(noise.cgm_obs_sigma_mgdl),
                "cgm_bias_mgdl": float(noise.cgm_bias_mgdl),
                "cgm_scale_bias": float(noise.cgm_scale_bias),
                "cgm_rw_sigma_bias": float(noise.cgm_rw_sigma_bias),
                "cgm_dropout_prob": float(noise.cgm_dropout_prob),
            },
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

    def after_step(self, observation, info: dict) -> tuple[Any, dict]:
        """Apply observation-level gaps and attach auditable sensor diagnostics."""
        if self._current_record is None or self._last_exposed_cgm is None:
            raise RuntimeError("after_step called before reset_episode")
        if self._controller_step >= len(self._gap_schedule):
            raise RuntimeError("controller produced more steps than the gap schedule")

        native_cgm = _scalar(
            info.get("native_cgm_mgdl", _observation_value(observation, 0))
        )
        true_bg = _scalar(info.get("true_bg_mgdl", np.nan))
        native_trend = _observation_value(observation, 3)
        injected_dropout = bool(self._gap_schedule[self._controller_step])

        if injected_dropout:
            exposed_cgm = float(self._last_exposed_cgm)
            exposed_trend = 0.0
            observation = _replace_observation_sensor(
                observation, exposed_cgm, exposed_trend
            )
            # Keep GlucoSim's own glucose-dependent action gates on the same
            # forward-filled trace seen by policy and Shield.
            state = dict(self.raw_env._jax_state)
            index = int(np.asarray(state["index"])) - 1
            state["CGM_hist"] = state["CGM_hist"].at[index].set(exposed_cgm)
            state["cgm_last"] = jnp.asarray(exposed_cgm, dtype=jnp.float32)
            state["cgm_trend"] = jnp.asarray(0.0, dtype=jnp.float32)
            self.raw_env._jax_state = state
            self._current_record["realized_markov_gap_steps"].append(
                self._controller_step
            )
        else:
            exposed_cgm = native_cgm
            exposed_trend = native_trend
            self._last_exposed_cgm = exposed_cgm

        scale = _scalar(
            info.get("cgm_scale", np.asarray(self.raw_env._jax_state["cgm_scale"]))
        )
        native_dropout_count = int(_scalar(info.get("native_dropout_count", 0)))
        sensor_clip_count = int(_scalar(info.get("sensor_clip_count", 0)))
        scale_boundary_count = int(
            _scalar(info.get("cgm_scale_boundary_count", 0))
        )
        self._current_record["native_dropout_minutes"] += native_dropout_count
        self._current_record["sensor_clip_minutes"] += sensor_clip_count
        self._current_record["cgm_scale_boundary_minutes"] += scale_boundary_count
        self._current_record["exposed_clip_observations"] += int(
            exposed_cgm <= 40.0 or exposed_cgm >= 400.0
        )
        self._current_record["cgm_scale_min"] = min(
            self._current_record["cgm_scale_min"], scale
        )
        self._current_record["cgm_scale_max"] = max(
            self._current_record["cgm_scale_max"], scale
        )
        self._current_record["cgm_scale_boundary_observations"] += int(
            scale <= 0.9 or scale >= 1.1
        )

        enriched_info = {
            **info,
            "true_bg_mgdl": true_bg,
            "native_cgm_mgdl": native_cgm,
            "exposed_cgm_mgdl": exposed_cgm,
            "exposed_cgm_trend_mgdl": exposed_trend,
            "cgm_scale": scale,
            "native_dropout_count": native_dropout_count,
            "sensor_clip_count": sensor_clip_count,
            "cgm_scale_boundary_count": scale_boundary_count,
            "injected_markov_dropout": injected_dropout,
            "exposed_sensor_clipped": bool(
                exposed_cgm <= 40.0 or exposed_cgm >= 400.0
            ),
        }
        self._controller_step += 1
        return observation, enriched_info

    def write_metadata(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / "stress_episode_metadata.json"
        payload = {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "stress_scenario": self.scenario,
            "requested_stress_scenario": self.requested_scenario,
            "stress_config": _jsonable(asdict(self.stress)),
            "base_evaluation_seed": self.base_seed,
            "replicate_id": self.replicate_id,
            "episode_seed_policy": (
                "base_evaluation_seed + replicate_id * 10000 + episode"
            ),
            "sensor_time_semantics": {
                "native_noise_update_minutes": 1,
                "controller_observation_minutes": self.sample_time,
                "controller_cgm_aggregation": "mean of native minute measurements",
                "markov_gap_layer": "controller observation",
            },
            "outcome_source": "true physiological BG_hist average",
            "unannounced_meal_semantics": (
                "inject pending carbohydrates after policy/Shield action selection "
                "and before env.step; never add the event to scenario_meals"
            ),
            "episodes": self.episode_records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path
