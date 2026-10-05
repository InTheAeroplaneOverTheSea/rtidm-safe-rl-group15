from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp

import glucosim
from glucosim import gym_env as gym


@dataclass
class StressConfig:
    """
    Configuration for simulator stress conditions.

    Defaults correspond to the nominal simulator.
    """

    # Meal timing stress
    meal_shift_minutes: float = 0.0

    # Sensor observation stress
    # None = use simulator default value
    cgm_noise_sigma: Optional[float] = None

    # Physiological OOD
    kabs_scale: float = 1.0


class GlucoSimInterface:
    """
    Unified interface for constructing GlucoSim environments
    under nominal or stressed conditions.
    """

    def __init__(
        self,
        patient_name: str = "adult#001",
        simulation_minutes: int = 1440,
        sample_time: int = 5,
    ):
        self.patient_name = patient_name
        self.simulation_minutes = simulation_minutes
        self.sample_time = sample_time

    # ==========================================================
    # Public API
    # ==========================================================

    def create(
        self,
        stress: Optional[StressConfig] = None,
        seed: int = 0,
    ):
        """
        Create and reset an environment with the requested stress.

        Returns:
            env
            obs
            info
        """

        if stress is None:
            stress = StressConfig()

        # 1. Create environment.
        #    Physiological parameters must be applied here.
        env = self._create_env(stress)

        # 2. Sensor configuration must be changed before reset.
        self._apply_sensor_stress(env, stress)

        # 3. Reset generates physiological state and meal scenario.
        obs, info = env.reset(seed=seed)

        # 4. Meal timing can only be modified after reset,
        #    because scenario_meals are generated during reset.
        self._apply_meal_stress(env, stress)

        return env, obs, info

    # ==========================================================
    # Environment construction / dynamics OOD
    # ==========================================================

    def _create_env(self, stress: StressConfig):
        patient_overrides = {}

        if stress.kabs_scale != 1.0:

            base_kabs = self._get_base_kabs()

            patient_overrides["kabs"] = (
                base_kabs * stress.kabs_scale
            )

        env = gym.make(
            "t1d-v0",
            simulation_minutes=self.simulation_minutes,
            sample_time=self.sample_time,
            patient_name=self.patient_name,
            patient_overrides=patient_overrides,
        )

        return env

    def _get_base_kabs(self):
        """
        Read nominal kabs for the selected patient.

        This avoids hardcoding adult#001's kabs value,
        so the interface also works for other patients.
        """

        probe_env = gym.make(
            "t1d-v0",
            simulation_minutes=self.simulation_minutes,
            sample_time=self.sample_time,
            patient_name=self.patient_name,
        )

        base_kabs = (
            probe_env.unwrapped
            .get_patient_params()
            .kabs
        )

        probe_env.close()

        return float(base_kabs)

    # ==========================================================
    # Sensor stress
    # ==========================================================

    def _apply_sensor_stress(
        self,
        env,
        stress: StressConfig,
    ):
        if stress.cgm_noise_sigma is None:
            return

        e = env.unwrapped

        new_noise_config = replace(
            e.env_params.noise_config,
            cgm_obs_sigma_mgdl=stress.cgm_noise_sigma,
        )

        e.env_params = replace(
            e.env_params,
            noise_config=new_noise_config,
        )

    # ==========================================================
    # Meal timing stress
    # ==========================================================

    def _apply_meal_stress(
        self,
        env,
        stress: StressConfig,
    ):
        shift = stress.meal_shift_minutes

        if shift == 0:
            return

        e = env.unwrapped

        scenario = e._jax_state["scenario_meals"]

        times = scenario[..., 0]
        amounts = scenario[..., 1]

        valid = (times >= 0) & (amounts > 0)

        shifted_times = jnp.where(
            valid,
            times + shift,
            times,
        )

        # For our current 24 h experiments,
        # do not allow meals to cross the day boundary.
        valid_shifted_times = shifted_times[valid]

        if valid_shifted_times.size > 0:

            max_time = float(
                jnp.max(valid_shifted_times)
            )

            if max_time >= 1440:
                raise ValueError(
                    f"Meal shift moves a meal outside "
                    f"the 24 h simulation window: "
                    f"{max_time:.1f} min"
                )

        shifted_scenario = (
            scenario
            .at[..., 0]
            .set(shifted_times)
        )

        # Update actual simulator state
        new_state = dict(e._jax_state)

        new_state["scenario_meals"] = (
            shifted_scenario
        )

        e._jax_state = new_state

        # Keep wrapper copy synchronized
        e.scenario_meals = shifted_scenario