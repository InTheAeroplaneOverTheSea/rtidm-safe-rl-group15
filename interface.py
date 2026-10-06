from dataclasses import dataclass, replace
from typing import Optional

import jax.numpy as jnp

import glucosim
from glucosim import gym_env as gym


@dataclass
class StressConfig:
    # Meal timing
    meal_shift_minutes: float = 0.0

    # Physiological OOD
    kabs_scale: float = 1.0

    # CGM sensor stress
    cgm_noise_sigma: Optional[float] = None
    # Standard deviation of zero-mean Gaussian noise added independently
    # to each CGM observation, in mg/dL.

    cgm_bias_mgdl: Optional[float] = None
    # Constant additive bias applied to all CGM measurements, in mg/dL.
    # Simulates persistent sensor calibration error.

    cgm_rw_sigma_bias: Optional[float] = None
    # Standard deviation of the random-walk increment of CGM sensor bias
    # at each simulation step. Simulates slowly drifting sensor bias.

    cgm_dropout_prob: Optional[float] = None 
    # Probability that a CGM measurement is dropped/missing at each step.
    # Value should typically be between 0 and 1
    # tips: Its effect may be limited because the simulator smooths/interpolates, so be brave to use a large number

    # Temporally correlated sensor noise
    cgm_use_ou: Optional[bool] = None # you can ignore this because it changes the some physiological mechanism parameter
    cgm_ou_sigma: Optional[float] = None # and in next update it will be moved to proper place


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
        e = env.unwrapped

        noise = e.env_params.noise_config

        updates = {}

        if stress.cgm_noise_sigma is not None:
            updates["cgm_obs_sigma_mgdl"] = (
                stress.cgm_noise_sigma
        )

        if stress.cgm_bias_mgdl is not None:
            updates["cgm_bias_mgdl"] = (
                stress.cgm_bias_mgdl
        )

        if stress.cgm_rw_sigma_bias is not None:
            updates["cgm_rw_sigma_bias"] = (
                stress.cgm_rw_sigma_bias
        )

        if stress.cgm_dropout_prob is not None:
            updates["cgm_dropout_prob"] = (
                stress.cgm_dropout_prob
        )

        if stress.cgm_use_ou is not None:
            updates["use_ou"] = (
                stress.cgm_use_ou
        )

        if stress.cgm_ou_sigma is not None:
            updates["ou_sigma_dL"] = (
                stress.cgm_ou_sigma
        )

        if not updates:
            return

        new_noise_config = replace(
            noise,
            **updates,
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