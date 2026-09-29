# Evaluation perturbations

Perturbations are project-owned transforms applied only to the controller's copy
of an observation. GlucoSim receives actions but never receives a corrupted state,
and safety metrics continue to use the raw post-step CGM trace.

## Configuration

Select a condition by passing one YAML or JSON file to
`experiments/evaluate_with_perturbations.py`:

```yaml
version: 1
name: cgm_bias_plus30
observation:
  - type: persistent_cgm_bias
    parameters:
      bias_mgdl: 30.0
```

`name` is recorded in result metadata and becomes the output-directory suffix.
Each entry in `observation` is applied in list order, so multiple transforms can be
composed without editing the evaluation loop. An empty list is the clean condition.

## Runtime lifecycle

At each episode reset, the pipeline receives the episode index and seed. At every
decision it receives a copy of the raw observation plus a context containing the
episode index, step index, and episode seed. This supports later stateful designs
such as random onset, bounded duration, drift, dropout, and seeded noise.

The controller CSV keeps `CGM` as the raw post-step value used for safety metrics
and `controller_CGM` as the pre-step value shown to the policy and shield.

## Adding a perturbation

1. Implement `ObservationPerturbation` in
   `src/rtidm_safe_rl/perturbations/observation.py`.
2. Put episode-state initialization in `reset()` and transformation logic in
   `apply()`; never modify the passed array in place.
3. Register the class in `OBSERVATION_PERTURBATIONS` in `config.py`.
4. Add a YAML condition under `configs/perturbations/` and unit tests under
   `tests/`.

No GlucoSim changes or perturbation-specific edits to `eval_run.py` are needed.
