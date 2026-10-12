## Changes: Predictor Output Logging

Added functionality to record and inspect predictor outputs during evaluation.

### 1. `GlucoAlg/shield/predictive_shield.py`

Added three attributes to track prediction-related information:

- `self.last_prediction`
- `self.last_action_pairs`
- `self.last_prediction_status`

### 2. `GlucoAlg/eval_run.py`

Modified the `evaluate_model()` function to capture and output predictor-related information during evaluation.

### 3. where is output?

in \GlucoAlg\diabetes_evaluation you will see a new json called `episode_0_forecasts.json`