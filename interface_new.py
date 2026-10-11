"""GlucoSim evaluation interface with reproducible stress scenarios.

Tested against GlucoSim revisions 6522d49ee1e6c1fd34f287519a5d41df735bbd38
and ddd674e35a32a3a6406a0c2d5e7bf3c72f56dfec.

The public ``create(stress, seed) -> (env, obs, info)`` API is preserved.
For an existing environment, use ``interface.reset(env, stress, seed)`` instead
of calling ``env.reset()`` directly: a direct reset replaces scenario_meals.

Scheduled-meal transformations are **announced** through GlucoSim's next-meal
observation fields. ``unannounced_daily_meals`` are different: call
``inject_unannounced_meals`` after the policy/Shield chooses its action but
before ``env.step``. The carbohydrates are then added to the real patient state
without ever appearing in the pre-event meal-look-ahead observation.
"""

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np

import glucosim  # noqa: F401 - registers local Gym IDs
from glucosim import gym_env as gym


@dataclass(frozen=True)
class Meal:
    """A planned meal. ``time`` is minutes since midnight or ``'HH:MM'``."""

    time: Union[float, str]
    carbs_g: float


MealInput = Union[Meal, tuple, dict]


@dataclass
class StressConfig:
    # Meal stress: applied to the ORIGINAL schedule unless overridden below.
    meal_shift_minutes: float = 0.0
    meal_amount_scale: float = 1.0
    # One schedule repeated on every evaluation day, e.g.
    # [Meal('07:00', 45), Meal('12:30', 70), Meal('18:00', 80)].
    # List length defines the number of meals per day; [] means no meals.
    daily_meals: Optional[Sequence[MealInput]] = None
    # An explicit schedule for EACH day (length must equal horizon days).
    # Mutually exclusive with daily_meals.
    meals_by_day: Optional[Sequence[Sequence[MealInput]]] = None
    # Independent seeded variability per meal; zero reproduces deterministic plans.
    meal_time_jitter_minutes: float = 0.0  # Gaussian standard deviation, minutes
    meal_amount_jitter_fraction: float = 0.0  # Gaussian relative std, e.g. 0.1
    max_meals_per_day: int = 10  # fixed JAX padded shape after reset
    # What to do with shifted meals before 0 or after evaluation horizon.
    meal_out_of_horizon: Literal["error", "drop"] = "error"

    # Surprise meals repeated on every evaluation day. These are deliberately
    # absent from scenario_meals and therefore from the pre-event look-ahead
    # observation. The evaluator must inject them immediately before env.step.
    # Example: [Meal("15:00", 60)] adds one hidden 60 g meal per day.
    unannounced_daily_meals: Optional[Sequence[MealInput]] = None

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


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number, got {value!r}")
    return value


def _parse_clock(value: Union[float, str]) -> float:
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"Meal time must be HH:MM, got {value!r}")
        try:
            hour, minute = map(int, parts)
        except ValueError as exc:
            raise ValueError(f"Invalid HH:MM meal time: {value!r}") from exc
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"Invalid HH:MM meal time: {value!r}")
        return float(hour * 60 + minute)
    minute = float(value)
    if not np.isfinite(minute) or not (0 <= minute < 1440):
        raise ValueError(f"Meal time must be in [0, 1440), got {minute!r}")
    return minute


def _parse_meal(raw: MealInput) -> tuple[float, float]:
    if isinstance(raw, Meal):
        time, carbs_g = raw.time, raw.carbs_g
    elif isinstance(raw, dict):
        # Allow both the paper-friendly 'time' and explicit 'time_minutes'.
        time = raw.get("time", raw.get("time_minutes"))
        carbs_g = raw.get("carbs_g")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        time, carbs_g = raw
    else:
        raise ValueError(f"Meal must be Meal, (time, carbs), or dict: {raw!r}")
    return _parse_clock(time), _finite_nonnegative(carbs_g, "carbs_g")


class GlucoSimInterface:
    """Create/reset GlucoSim environments with physiological, CGM, and meal stress."""

    def __init__(
        self,
        patient_name: str = "adult#001",
        simulation_minutes: int = 1440,
        sample_time: int = 5,
    ):
        # GlucoSim's multiday generator uses simulation_minutes // 1440, and
        # step() indexes one schedule per calendar day. Reject partial days to
        # prevent a missing final-day schedule in this pinned revision.
        if int(simulation_minutes) != simulation_minutes or simulation_minutes < 1440 or simulation_minutes % 1440:
            raise ValueError("simulation_minutes must be a positive whole number of days (1440 * days)")
        if int(sample_time) != sample_time or sample_time <= 0 or 1440 % sample_time:
            raise ValueError("sample_time must be a positive integer dividing 1440")
        self.patient_name = patient_name
        self.simulation_minutes = int(simulation_minutes)
        self.sample_time = int(sample_time)
        self._delivered_unannounced: dict[int, list[dict]] = {}

    @property
    def horizon_days(self) -> int:
        return self.simulation_minutes // 1440

    def create(self, stress: Optional[StressConfig] = None, seed: int = 0):
        """Construct and reset. Returns ``(env, corrected_obs, info)``.

        The environment object is not automatically wrapped. For EACH later
        evaluation episode, call ``interface.reset(env, stress, seed)`` so the
        meal schedule is re-applied after GlucoSim regenerates it.
        """
        stress = stress if stress is not None else StressConfig()
        self._validate_meal_stress(stress)
        env = self._create_env(stress)
        self.configure_existing_env(env, stress)
        try:
            obs, info = self.reset(env, stress, seed)
        except Exception:
            env.close()
            raise
        return env, obs, info

    def reset(self, env, stress: Optional[StressConfig] = None, seed: int = 0):
        """Re-reset the SAME env and re-apply meal stress each episode."""
        stress = stress if stress is not None else StressConfig()
        self._validate_meal_stress(stress)
        obs, info = env.reset(seed=seed)
        corrected = self.after_reset(env, stress, seed)
        if corrected is not None:
            obs = corrected
        return obs, info

    def configure_existing_env(self, env, stress: Optional[StressConfig] = None):
        """Apply pre-reset stress to an already constructed Gym environment.

        This is the entry point used by the author-v1 evaluator, which owns the
        ``DiabetesEnvs`` wrapper and therefore cannot use :meth:`create`.
        ``kabs_scale`` must still be supplied at construction time.
        """
        stress = stress if stress is not None else StressConfig()
        self._validate_meal_stress(stress)
        if stress.kabs_scale != 1.0:
            raise ValueError(
                "kabs_scale cannot be applied to an existing environment; "
                "construct the environment through GlucoSimInterface.create"
            )
        self._apply_sensor_stress(env, stress)

    def after_reset(self, env, stress: Optional[StressConfig] = None, seed: int = 0):
        """Apply post-reset schedule changes and start a new hidden-meal log.

        Returns a corrected initial observation for announced schedule changes,
        or ``None`` when the reset observation is already correct.
        """
        stress = stress if stress is not None else StressConfig()
        self._validate_meal_stress(stress)
        self._delivered_unannounced[id(env.unwrapped)] = []
        return self._apply_meal_stress(env, stress, seed)

    def planned_unannounced_meal_events(
        self, stress: Optional[StressConfig] = None
    ) -> list[dict]:
        """Return the hidden event plan without exposing it to the environment."""
        stress = stress if stress is not None else StressConfig()
        self._validate_meal_stress(stress)
        daily = stress.unannounced_daily_meals
        if daily is None:
            return []
        parsed = [_parse_meal(meal) for meal in daily]
        return [
            {
                "day": day + 1,
                "time_minutes": int(time),
                "absolute_time_minutes": day * 1440 + int(time),
                "carbs_g": float(carbs),
            }
            for day in range(self.horizon_days)
            for time, carbs in parsed
            if carbs > 0
        ]

    def inject_unannounced_meals(
        self, env, stress: Optional[StressConfig] = None
    ) -> list[dict]:
        """Inject due surprise meals after action selection and before ``step``.

        The meal is added directly to the patient's pending carbohydrate buffer,
        not to ``scenario_meals``. Consequently the current observation and all
        earlier meal-look-ahead observations cannot reveal it. Calling this
        method twice at the same simulator minute is idempotent.
        """
        stress = stress if stress is not None else StressConfig()
        self._validate_meal_stress(stress)
        if stress.unannounced_daily_meals is None:
            return []

        e = env.unwrapped
        if e._jax_state is None:
            raise RuntimeError("Unannounced meals can only be injected after reset")
        current_time = int(np.asarray(e._jax_state["t"]))
        env_key = id(e)
        delivered = self._delivered_unannounced.setdefault(env_key, [])
        delivered_times = {
            int(event["absolute_time_minutes"]) for event in delivered
        }
        if current_time in delivered_times:
            return []

        due = [
            event
            for event in self.planned_unannounced_meal_events(stress)
            if event["absolute_time_minutes"] == current_time
        ]
        if not due:
            return []

        total_carbs = sum(float(event["carbs_g"]) for event in due)
        new_state = dict(e._jax_state)
        pending = jnp.asarray(new_state["planned_meal"], dtype=jnp.float32)
        new_state["planned_meal"] = pending + jnp.asarray(
            total_carbs, dtype=jnp.float32
        )
        e._jax_state = new_state
        delivered.extend(dict(event) for event in due)
        return [dict(event) for event in due]

    def delivered_unannounced_meal_events(self, env) -> list[dict]:
        """Return a copy of the surprise meals actually injected this episode."""
        return [
            dict(event)
            for event in self._delivered_unannounced.get(id(env.unwrapped), [])
        ]

    def _create_env(self, stress: StressConfig):
        patient_overrides = {}
        if stress.kabs_scale != 1.0:
            patient_overrides["kabs"] = self._get_base_kabs() * stress.kabs_scale
        return gym.make(
            "t1d-v0",
            simulation_minutes=self.simulation_minutes,
            sample_time=self.sample_time,
            patient_name=self.patient_name,
            patient_overrides=patient_overrides,
        )

    def _get_base_kabs(self):
        probe_env = gym.make(
            "t1d-v0",
            simulation_minutes=self.simulation_minutes,
            sample_time=self.sample_time,
            patient_name=self.patient_name,
        )
        try:
            return float(probe_env.unwrapped.get_patient_params().kabs)
        finally:
            probe_env.close()

    def _apply_sensor_stress(self, env, stress: StressConfig):
        e = env.unwrapped
        noise = e.env_params.noise_config
        updates = {}
        if stress.cgm_noise_sigma is not None:
            updates["cgm_obs_sigma_mgdl"] = _finite_nonnegative(stress.cgm_noise_sigma, "cgm_noise_sigma")
        if stress.cgm_bias_mgdl is not None:
            updates["cgm_bias_mgdl"] = float(stress.cgm_bias_mgdl)
        if stress.cgm_rw_sigma_bias is not None:
            updates["cgm_rw_sigma_bias"] = _finite_nonnegative(stress.cgm_rw_sigma_bias, "cgm_rw_sigma_bias")
        if stress.cgm_dropout_prob is not None:
            p = float(stress.cgm_dropout_prob)
            if not 0 <= p <= 1:
                raise ValueError("cgm_dropout_prob must be between 0 and 1")
            updates["cgm_dropout_prob"] = p
        if stress.cgm_use_ou is not None:
            updates["use_ou"] = bool(stress.cgm_use_ou)
        if stress.cgm_ou_sigma is not None:
            updates["ou_sigma_dL"] = _finite_nonnegative(stress.cgm_ou_sigma, "cgm_ou_sigma")
        if updates:
            e.env_params = replace(e.env_params, noise_config=replace(noise, **updates))

    @staticmethod
    def _meal_stress_enabled(stress: StressConfig) -> bool:
        return (
            stress.daily_meals is not None
            or stress.meals_by_day is not None
            or stress.meal_shift_minutes != 0
            or stress.meal_amount_scale != 1.0
            or stress.meal_time_jitter_minutes != 0
            or stress.meal_amount_jitter_fraction != 0
        )

    def _validate_meal_stress(self, stress: StressConfig):
        if stress.daily_meals is not None and stress.meals_by_day is not None:
            raise ValueError("Use only one of daily_meals and meals_by_day")
        if not np.isfinite(stress.meal_shift_minutes):
            raise ValueError("meal_shift_minutes must be finite")
        _finite_nonnegative(stress.meal_amount_scale, "meal_amount_scale")
        _finite_nonnegative(stress.meal_time_jitter_minutes, "meal_time_jitter_minutes")
        _finite_nonnegative(stress.meal_amount_jitter_fraction, "meal_amount_jitter_fraction")
        if isinstance(stress.max_meals_per_day, bool) or not isinstance(stress.max_meals_per_day, int) or stress.max_meals_per_day < 1:
            raise ValueError("max_meals_per_day must be a positive integer")
        if stress.meal_out_of_horizon not in ("error", "drop"):
            raise ValueError("meal_out_of_horizon must be 'error' or 'drop'")
        if stress.unannounced_daily_meals is not None:
            for meal in stress.unannounced_daily_meals:
                time, _ = _parse_meal(meal)
                if not float(time).is_integer() or int(time) % self.sample_time:
                    raise ValueError(
                        "Unannounced meal times must be integer multiples of "
                        f"sample_time={self.sample_time}, got {time}"
                    )

    def _original_or_custom_meals(self, original: np.ndarray, stress: StressConfig):
        """Return a list of (day, minutes_after_midnight, carbohydrate_g)."""
        days = original.shape[0]
        if stress.daily_meals is not None:
            one_day = [_parse_meal(meal) for meal in stress.daily_meals]
            return [(day, t, a) for day in range(days) for t, a in one_day if a > 0]
        if stress.meals_by_day is not None:
            if len(stress.meals_by_day) != days:
                raise ValueError(f"meals_by_day must provide {days} days, got {len(stress.meals_by_day)}")
            return [
                (day, t, a)
                for day, meals in enumerate(stress.meals_by_day)
                for t, a in (_parse_meal(meal) for meal in meals)
                if a > 0
            ]
        # Preserve the existing sampled meal profile (only actual in-day meals).
        return [
            (day, float(t), float(a))
            for day in range(days)
            for t, a in original[day]
            if np.isfinite(t) and np.isfinite(a) and 0 <= t < 1440 and a > 0
        ]

    def build_meal_schedule(
        self,
        original_scenario: np.ndarray,
        stress: StressConfig,
        seed: int = 0,
    ) -> np.ndarray:
        """Pure schedule transformation: output [days, max_meals_per_day, 2].

        Each row is [minute_in_day, carbs_g]. [-1, 0] marks an empty slot.
        Meals shifted across midnight move into the correct adjacent day.
        Stochastic jitters use their own NumPy RNG so shield/no-shield can
        share the EXACT SAME planned stress schedule under the same seed.
        """
        self._validate_meal_stress(stress)
        original = np.asarray(original_scenario, dtype=np.float64)
        if original.ndim != 3 or original.shape[-1] != 2:
            raise ValueError(f"Expected scenario shape [days, slots, 2], got {original.shape}")
        if original.shape[0] != self.horizon_days:
            raise ValueError(f"Simulator generated {original.shape[0]} days; expected {self.horizon_days}")

        out = np.zeros((self.horizon_days, stress.max_meals_per_day, 2), dtype=np.float32)
        out[..., 0] = -1
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 101031]))
        events = [[] for _ in range(self.horizon_days)]
        horizon = self.simulation_minutes

        for day, minute, carbs in self._original_or_custom_meals(original, stress):
            new_time = (day * 1440 + minute + float(stress.meal_shift_minutes))
            if stress.meal_time_jitter_minutes:
                new_time += rng.normal(0, stress.meal_time_jitter_minutes)
            carbs *= float(stress.meal_amount_scale)
            if stress.meal_amount_jitter_fraction:
                carbs *= max(0.0, 1.0 + rng.normal(0, stress.meal_amount_jitter_fraction))
            carbs = max(0.0, carbs)
            if carbs == 0:
                continue
            # mini_step compares round(meal_time) against current minute.
            new_time = int(np.floor(new_time + 0.5))
            if not 0 <= new_time < horizon:
                if stress.meal_out_of_horizon == "drop":
                    continue
                raise ValueError(
                    f"Shifted meal on day {day + 1} lands outside "
                    f"{self.horizon_days}-day horizon at minute {new_time}; "
                    "use meal_out_of_horizon='drop' to intentionally omit it."
                )
            new_day, time_within_day = divmod(new_time, 1440)
            events[new_day].append((time_within_day, carbs))

        for day, rows in enumerate(events):
            # Two events in the same minute combine: this matches step()'s sum.
            by_time: dict[int, float] = {}
            for t, a in rows:
                by_time[t] = by_time.get(t, 0.0) + a
            sorted_rows = sorted(by_time.items())
            if len(sorted_rows) > stress.max_meals_per_day:
                raise ValueError(
                    f"Day {day + 1} has {len(sorted_rows)} meals, "
                    f"greater than max_meals_per_day={stress.max_meals_per_day}"
                )
            for slot, (time, grams) in enumerate(sorted_rows):
                out[day, slot] = (time, grams)
        return out

    def _apply_meal_stress(self, env, stress: StressConfig, seed: int):
        if not self._meal_stress_enabled(stress):
            return None
        e = env.unwrapped
        if e._jax_state is None:
            raise RuntimeError("Meal stress must be applied AFTER reset")
        scenario = self.build_meal_schedule(
            np.asarray(e._jax_state["scenario_meals"]), stress, seed
        )
        new_state = dict(e._jax_state)
        new_state["scenario_meals"] = jnp.asarray(scenario, dtype=jnp.float32)
        e._jax_state = new_state
        e.scenario_meals = new_state["scenario_meals"]

        # The initial observation contains time-to-next-meal and meal size.
        # Rebuild it from the new state; otherwise the first action sees stale
        # meal information. This is the original GlucoSim function.
        from glucosim.simglucose.rl.observation_space import _build_observation

        return np.asarray(_build_observation(new_state, e.env_params), dtype=np.float32)

    @staticmethod
    def get_meal_schedule(env) -> np.ndarray:
        """Copy the actual planned schedule for logging and paired comparisons."""
        return np.array(env.unwrapped._jax_state["scenario_meals"], copy=True)

    @staticmethod
    def get_meal_events(env) -> list[dict]:
        """Serializable scheduled events; does not imply every meal was eaten."""
        events = []
        schedule = GlucoSimInterface.get_meal_schedule(env)
        for day, rows in enumerate(schedule):
            for t, g in rows:
                if t >= 0 and g > 0:
                    events.append({"day": day + 1, "time_minutes": float(t), "carbs_g": float(g)})
        return events
