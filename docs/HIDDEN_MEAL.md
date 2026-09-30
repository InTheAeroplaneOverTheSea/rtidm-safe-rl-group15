# Hidden-meal smoke comparison

Copy the ZIP contents into the group repository root. These are new project-owned
files. The patch adds two small hooks to the group's GlucoSim step implementation;
it does not change the shield or checkpoint files.

From the repository root in Windows PowerShell:

```powershell
git -C external/GlucoSim apply ../../patches/hidden_meal.patch
.\.venv\Scripts\python.exe .\experiments\evaluate_hidden_meal.py
```

The runner disables TorchDynamo compilation on Windows unless already configured,
and keeps JAX on CPU. It uses the same CPO seed-0 epoch-2441 and BA-NODE seed-42
models as the group's smoke test, patient adolescent#010, eval seed 22, upper
threshold 170, penalty 10, top-k 2, and 20 context examples.

Four conditions each simulate one seven-day episode. One additional 60 g meal
starts at minute 720 (12 hours from episode start), once per episode. The simulator
uses its normal ingestion/absorption mechanics. The meal is added after reset to
a separate minute-resolution schedule, bypasses controller meal acceptance, and
is not inserted into scenario_meals or the announced upcoming-meal preview.
After ingestion its effects can be seen through physiology, including glucose
and carbohydrates on board. This is an extra meal, not a replacement or concealment
of an existing scheduled meal.

Clean runs receive a zero schedule for matching JAX state structure. Observations
and sampled actions must match within each shield setting through the controller
decision at minute 720, before the first affected simulator step. An episode that
ends before ingestion fails the check. This check is performed after the four runs.

Results are copied into results/hidden_meal/{no_shield,shield}__{clean,meal}/,
including the group's detailed/summary results and controller CSVs, plus explicit
meal event metadata. results/hidden_meal/summary.csv combines the four summaries;
experiment.json records parameters and a successful premeal comparison.
The upstream evaluator also retains its usual outputs beneath
external/GlucoAlg/diabetes_evaluation/ with unique condition/run suffixes.
The upstream meal_total_g/controller recommendation fields do not count the extra
external meal; use physiology_events.json for its amount and scheduled onset.

For a fresh repeat, change the output directory:

```powershell
.\.venv\Scripts\python.exe .\experiments\evaluate_hidden_meal.py --output-dir results/hidden_meal_repeat
```

More seeds, patients, meal sizes, and repeated daily perturbations should follow
a frozen group validation protocol. The default 170 threshold is only the group's
smoke setting, not a calibrated paper reproduction. One episode is a functional
check, not a statistical safety finding.

Validation performed here: patch applicability to GlucoSim c3c40a2, Python syntax,
meal addition with scenario/controller overlap, preservation through patient_step,
unchanged announced schedule, reset isolation, transparent wrapper attributes,
and rejection of mismatched premeal actions. These used isolated function/wrapper
checks; the complete trained-policy/JAX rollout was not executed here.
