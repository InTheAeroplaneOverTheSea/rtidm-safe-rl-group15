from __future__ import annotations

import unittest
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import torch

from interface_new import GlucoSimInterface, StressConfig
from scripts.analyze_author_v1_cpo_stress_calibration import (
    expected_effective_noise,
    expected_stress_fields,
    planned_gap_steps,
)
from scripts.author_v1_stress_hook import (
    AuthorV1StressController,
    _episode_seed,
    _gap_intervals,
    canonical_scenario_name,
)


@dataclass(frozen=True)
class _NoiseConfig:
    cgm_obs_sigma_mgdl: float = 1.5
    cgm_bias_mgdl: float = 0.0
    cgm_scale_bias: float = 0.0
    cgm_rw_sigma_bias: float = 0.002
    cgm_dropout_prob: float = 0.01
    use_ou: bool = False
    ou_sigma_dL: float = 0.0


@dataclass(frozen=True)
class _EnvParams:
    noise_config: _NoiseConfig = _NoiseConfig()


class _RawEnv:
    def __init__(self) -> None:
        self.env_params = _EnvParams()
        self._jax_state = {
            "index": jnp.asarray(1),
            "CGM_hist": jnp.asarray([90.0], dtype=jnp.float32),
            "cgm_last": jnp.asarray(90.0, dtype=jnp.float32),
            "cgm_trend": jnp.asarray(2.0, dtype=jnp.float32),
            "cgm_scale": jnp.asarray(1.0, dtype=jnp.float32),
        }

    @property
    def unwrapped(self):
        return self


class InterfaceSensorConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interface = GlucoSimInterface(simulation_minutes=1440, sample_time=5)

    def test_applies_finite_native_sensor_overrides(self) -> None:
        env = _RawEnv()
        self.interface.configure_existing_env(
            env,
            StressConfig(
                cgm_noise_sigma=15.0,
                cgm_bias_mgdl=-10.0,
                cgm_rw_sigma_bias=0.003,
                cgm_dropout_prob=0.2,
            ),
        )

        noise = env.env_params.noise_config
        self.assertEqual(noise.cgm_obs_sigma_mgdl, 15.0)
        self.assertEqual(noise.cgm_bias_mgdl, -10.0)
        self.assertEqual(noise.cgm_rw_sigma_bias, 0.003)
        self.assertEqual(noise.cgm_dropout_prob, 0.2)

    def test_rejects_nonfinite_bias(self) -> None:
        with self.assertRaisesRegex(ValueError, "cgm_bias_mgdl must be finite"):
            self.interface.configure_existing_env(
                _RawEnv(), StressConfig(cgm_bias_mgdl=float("nan"))
            )

    def test_markov_gap_fields_are_atomic_and_match_sample_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be set together"):
            self.interface.configure_existing_env(
                _RawEnv(), StressConfig(cgm_gap_entry_prob=0.003)
            )

        self.interface.configure_existing_env(
            _RawEnv(),
            StressConfig(
                cgm_gap_entry_prob=0.003,
                cgm_gap_recovery_prob=0.2917,
                cgm_gap_sampling_interval_min=np.int64(5),
            ),
        )

        with self.assertRaisesRegex(ValueError, "must match"):
            self.interface.configure_existing_env(
                _RawEnv(),
                StressConfig(
                    cgm_gap_entry_prob=0.003,
                    cgm_gap_recovery_prob=0.2917,
                    cgm_gap_sampling_interval_min=1,
                ),
            )


class StressControllerTests(unittest.TestCase):
    def test_alias_and_literal_episode_seed_contract(self) -> None:
        self.assertEqual(canonical_scenario_name("sensor_sigma15"), "gaussian_sigma15")
        self.assertEqual([_episode_seed(22, 0, i) for i in range(3)], [22, 23, 24])
        self.assertEqual(_episode_seed(22, 2, 3), 20_025)
        with self.assertRaisesRegex(ValueError, "smaller than"):
            _episode_seed(22, 0, 10_000)

    def test_gap_intervals_round_trip_to_steps(self) -> None:
        schedule = np.asarray([False, True, True, False, True], dtype=np.bool_)
        intervals = _gap_intervals(schedule, sample_time=5)

        self.assertEqual(planned_gap_steps(intervals), [1, 2, 4])
        self.assertEqual(planned_gap_steps(intervals, horizon_steps=2), [1])
        self.assertEqual(intervals[0]["start_time_minutes"], 10)
        self.assertEqual(intervals[0]["duration_minutes"], 10)

    def test_markov_schedule_is_deterministic_and_seed_specific(self) -> None:
        controller = object.__new__(AuthorV1StressController)
        controller.stress = StressConfig(
            cgm_gap_entry_prob=0.2,
            cgm_gap_recovery_prob=0.4,
            cgm_gap_sampling_interval_min=5,
        )
        controller.max_controller_steps = 200

        first = controller._build_gap_schedule(22)
        repeated = controller._build_gap_schedule(22)
        different = controller._build_gap_schedule(23)

        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, different))

    def test_after_step_forward_fills_gap_for_policy_and_raw_state(self) -> None:
        controller = object.__new__(AuthorV1StressController)
        controller.raw_env = _RawEnv()
        controller._current_record = {
            "realized_markov_gap_steps": [],
            "native_dropout_minutes": 0,
            "sensor_clip_minutes": 0,
            "cgm_scale_boundary_minutes": 0,
            "exposed_clip_observations": 0,
            "cgm_scale_min": 1.0,
            "cgm_scale_max": 1.0,
            "cgm_scale_boundary_observations": 0,
        }
        controller._last_exposed_cgm = 123.0
        controller._gap_schedule = np.asarray([True], dtype=np.bool_)
        controller._controller_step = 0
        observation = torch.tensor([[95.0, 0.0, 0.0, -2.0]], dtype=torch.float32)
        info = {
            "true_bg_mgdl": np.asarray([100.0]),
            "native_cgm_mgdl": np.asarray([95.0]),
            "cgm_scale": np.asarray([1.0]),
            "native_dropout_count": np.asarray([1]),
            "sensor_clip_count": np.asarray([0]),
            "cgm_scale_boundary_count": np.asarray([0]),
        }

        exposed, enriched = controller.after_step(observation, info)

        self.assertEqual(float(exposed[0, 0]), 123.0)
        self.assertEqual(float(exposed[0, 3]), 0.0)
        self.assertEqual(float(controller.raw_env._jax_state["CGM_hist"][0]), 123.0)
        self.assertEqual(float(controller.raw_env._jax_state["cgm_last"]), 123.0)
        self.assertEqual(float(controller.raw_env._jax_state["cgm_trend"]), 0.0)
        self.assertTrue(enriched["injected_markov_dropout"])
        self.assertEqual(controller._current_record["realized_markov_gap_steps"], [0])
        self.assertEqual(controller._current_record["native_dropout_minutes"], 1)


class CalibrationContractTests(unittest.TestCase):
    def test_primary_scenario_contracts(self) -> None:
        self.assertEqual(
            expected_effective_noise("gaussian_sigma15")["cgm_obs_sigma_mgdl"],
            15.0,
        )
        self.assertEqual(
            expected_effective_noise("bias_negative")["cgm_bias_mgdl"], -10.0
        )
        self.assertEqual(
            expected_effective_noise("random_walk_drift")["cgm_rw_sigma_bias"],
            0.003,
        )
        self.assertEqual(
            expected_stress_fields("dropout_markov_v1"),
            {
                "cgm_noise_sigma": None,
                "cgm_bias_mgdl": None,
                "cgm_rw_sigma_bias": None,
                "cgm_dropout_prob": None,
                "cgm_gap_entry_prob": 0.003,
                "cgm_gap_recovery_prob": 0.2917,
                "cgm_gap_sampling_interval_min": 5,
            },
        )


if __name__ == "__main__":
    unittest.main()
