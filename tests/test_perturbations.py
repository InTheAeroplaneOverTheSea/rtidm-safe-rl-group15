import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "external" / "GlucoAlg"))

from eval_run import _count_excursions, evaluate_model  # noqa: E402
from rtidm_safe_rl.perturbations import (  # noqa: E402
    ObservationPerturbationPipeline,
    PersistentCGMBias,
    build_perturbation_pipeline,
    load_perturbation_pipeline,
)


class _FixedDistribution:
    def sample(self):
        return torch.tensor([[0, 0]], dtype=torch.int64)


class _RecordingActor:
    def __init__(self):
        self.obs_dim = 4
        self.shield = None
        self.normalized_inputs = []
        self.shield_inputs = []

    def __call__(self, obs, original_obs):
        self.normalized_inputs.append(obs.detach().cpu().numpy().copy())
        self.shield_inputs.append(original_obs.detach().cpu().numpy().copy())
        return _FixedDistribution()


class _ThreeStepEnvironment:
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([2, 2])
    sample_time = 5
    env = None

    def __init__(self):
        self.step_number = 0

    def reset(self, seed=None):
        del seed
        self.step_number = 0
        return np.array([[100.0, 1.0, 2.0, 3.0]], dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_number += 1
        cgm = 100.0 + self.step_number
        observation = np.array([[cgm, 1.0, 2.0, 3.0]], dtype=np.float32)
        terminated = np.array([self.step_number == 3])
        info = {"cgm": np.array([cgm]), "iob": 1.0, "cob": 2.0}
        return observation, np.array([0.0]), np.array([0.0]), terminated, np.array([False]), info


class PerturbationConfigTests(unittest.TestCase):
    def test_loads_plus_30_config(self):
        pipeline = load_perturbation_pipeline(
            REPOSITORY_ROOT / "configs" / "perturbations" / "cgm_bias_plus30.yaml"
        )
        pipeline.reset(episode_index=0, seed=22)
        raw = np.array([100.0, 1.0, 2.0, 3.0], dtype=np.float32)
        corrupted = pipeline.apply(raw, episode_index=0, step_index=0)
        np.testing.assert_array_equal(raw, [100.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(corrupted, [130.0, 1.0, 2.0, 3.0])
        self.assertEqual(pipeline.output_tag, "cgm_bias_plus30")

    def test_pipeline_composes_perturbations_in_order(self):
        pipeline = ObservationPerturbationPipeline(
            name="combined_bias",
            perturbations=[PersistentCGMBias(30), PersistentCGMBias(-5)],
        )
        pipeline.reset(episode_index=0, seed=22)
        corrupted = pipeline.apply(
            np.array([100.0, 1.0], dtype=np.float32),
            episode_index=0,
            step_index=0,
        )
        np.testing.assert_array_equal(corrupted, [125.0, 1.0])

    def test_unknown_perturbation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown observation perturbation"):
            build_perturbation_pipeline(
                {"version": 1, "name": "bad", "observation": [{"type": "unknown"}]}
            )


class EvaluationHookTests(unittest.TestCase):
    def test_excursions_are_not_counted_as_violating_samples(self):
        values = [100.0, 65.0, 60.0, 75.0, 190.0, 200.0, 170.0, 190.0]
        self.assertEqual(_count_excursions(values, lambda value: value < 70.0), 1)
        self.assertEqual(_count_excursions(values, lambda value: value > 180.0), 2)

    def test_controller_sees_bias_but_metrics_and_trace_remain_raw(self):
        perturbation = load_perturbation_pipeline(
            REPOSITORY_ROOT / "configs" / "perturbations" / "cgm_bias_plus30.yaml"
        )
        actor = _RecordingActor()
        env = _ThreeStepEnvironment()
        previous_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                os.chdir(temp_dir)
                results = evaluate_model(
                    env,
                    actor,
                    num_episodes=1,
                    seed=22,
                    patient_type="t1d",
                    algorithm="bias_test",
                    patient_name="adolescent#001",
                    save_seed=22,
                    shield_type="none",
                    observation_perturbation=perturbation,
                )

                controller_cgm = [float(value[0, 0]) for value in actor.shield_inputs]
                self.assertEqual(controller_cgm, [130.0, 131.0, 132.0])
                self.assertEqual(results["mean_glucose"], 102.0)
                self.assertEqual(results["mean_time_below_70_pct"], 0.0)
                self.assertEqual(results["mean_time_above_180_pct"], 0.0)
                self.assertEqual(results["mean_time_above_250_pct"], 0.0)
                self.assertEqual(results["total_hypo_excursions"], 0)
                self.assertEqual(results["total_hyper_excursions_180"], 0)
                self.assertEqual(results["perturbation"]["name"], "cgm_bias_plus30")

                trace_path = Path(
                    "diabetes_evaluation/t1d/bias_test/adolescent#001/seed22/"
                    "no_shield_cgm_bias_plus30/episode_0_controller.csv"
                )
                with trace_path.open(newline="") as trace_file:
                    rows = list(csv.DictReader(trace_file))
                self.assertEqual([float(row["CGM"]) for row in rows], [101.0, 102.0, 103.0])
                self.assertEqual(
                    [float(row["controller_CGM"]) for row in rows],
                    [130.0, 131.0, 132.0],
                )
        finally:
            os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
