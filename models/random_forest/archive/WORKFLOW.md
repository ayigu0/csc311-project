# Historical RF workflow (archived)

These commands describe the pre-archive layout. For historical reproduction, use
a separate checkout of commit `f62c0e6135d7ab87417543287e766b86895c186d`; do not run these commands
from this archive directory. Current commands are in [../README.md](../README.md).



Generate the versioned dataset, then run from the project root:

```bash
python models/random_forest/data_preprocessing.py
python models/random_forest/random_forest.py
```

RF now reads `models/random_forest/data/rf_features_v1.csv` by default.
It contains all 870 rows, the original student split (seed 42, 50%/20%/30%),
326 numeric features, and four metadata columns: `source_row_index`, `id`,
`label`, and `split`. Those four columns never enter the model.

`data_preprocessing.py` processes all 12 survey questions:

- Normalize Unicode, whitespace and capitalization; recognize exact unknown
  markers without removing meaningful phrases or bare `none` in associations.
- Validate ease scores in 0–10 and map Likert responses to 1–5. Invalid or
  missing numbers use training medians (zero for all-missing training columns).
- Multi-hot encode the fixed location, season and music categories, with
  separate missing-answer and unknown-option indicators.
- Parse ticket prices as CAD, including currency symbols, thousands separators,
  `k` suffixes and simple written amounts. A single range uses its midpoint.
  Explicit unwillingness is zero; ambiguous multiple amounts and foreign
  currencies become missing. Bare numbers/$ assume CAD. Large valid prices
  are retained, not clipped.
- Encode food/drinks, colour, song and feelings as binary word features.
  Each field uses up to 80 words appearing in at least two training responses.
  Unknown words in later data are ignored. No scaling is applied.
- Retain all rows, including empty responses; add missing indicators.

Only training rows determine medians and vocabularies. Frozen rules are saved
in `models/random_forest/data/rf_preprocessing_v1.json` with source/code/data
hashes. Loading refuses stale or modified data and asks for regeneration.
The original CSV and shared exploration code are unchanged. The saved CSV is
specific to this split: future cross-validation must refit preprocessing inside
each fold from raw data. This feature set extends the earlier eight-feature
baseline, so score changes reflect new features as well as cleaning.

For future label-free inputs, load the JSON and call
`transform_features(raw_dataframe, metadata["state"])` from the preprocessing
module. It preserves input row order and produces the saved feature columns.

The default forest has 100 trees, no depth limit, `min_samples_leaf=1`,
`max_features="sqrt"`, and random seed 42. These settings are not tuned.
Optional flags include `--n-estimators`, `--max-depth`, `--min-samples-leaf`,
`--min-samples-split`, `--max-features` (`sqrt`, `log2`, or a fraction in (0, 1]),
`--seed` (forest only), and `--output-root`.

The default split criterion is Gini impurity. To run the entropy version, or
compare both criteria with all other settings fixed:

```bash
python models/random_forest/random_forest.py --criterion entropy
python models/random_forest/compare_random_forest.py
```

The comparison saves `comparison.csv`, paired validation predictions, and a
separate full run folder for each criterion under
`results/random_forest_comparison_<UTC timestamp>/`. It verifies matching data,
preprocessing, splits, and all other model parameters. This is a single-seed
baseline comparison, not evidence that one criterion consistently wins.

Each baseline run creates a separate `results/random_forest_<criterion>_<UTC timestamp>/` folder
containing training/validation accuracy and macro-F1, the validation confusion
matrix, per-class report, validation predictions, and configuration metadata.
The test set is reserved and not evaluated. This script uses sklearn for
development; it is not the final restricted-library `pred.py` submission.

Check price parsing, row alignment, missing-value handling and frozen training
rules with `python -m unittest models.random_forest.test_data_preprocessing -v`.

To evaluate both fixed baselines on training, validation **and the local test
set**, run:

```bash
python models/random_forest/evaluate_random_forest.py
```

The evaluation checks student separation and saved split sizes, fits only the
435 training rows, selects a criterion by validation macro-F1 (accuracy breaks
ties), and then scores both models on the 261 local test rows. The validation
set has 174 rows. It saves a combined metrics table, per-class reports,
confusion matrices, predictions, and configuration under
`results/random_forest_evaluation_<UTC timestamp>/`. This explicitly exposes
local test performance; use the validation-only comparison for future tuning.
The local student test set is separate from the course's hidden TA/instructor
test set.

### RF hyperparameter tuning

```bash
python models/random_forest/tune_random_forest.py
python models/random_forest/random_forest.py --config models/random_forest/tuned_params.json
```

The bounded search uses the saved training/validation split and never evaluates
test predictions. Stage 1 compares 18 fixed structural configurations for each
of Gini and entropy, including the original baseline. The candidates are sampled
with seed 42 from depths `[5, 10, 20, None]`, leaf sizes `[1, 2, 5]`, and split
minimums `[2, 5, 10]`, using 100 trees and `sqrt` feature sampling. Stage 2 varies
`max_features` over `sqrt`, `log2`, and `0.3`, and tree count over 100 and 300 for
each criterion's structural winner. Duplicate configurations are skipped,
yielding 46 fits in total. Selection maximizes validation macro-F1, then
accuracy; ties prefer fewer trees, smaller trees, and the earlier trial.

The script saves every trial, its validation predictions, a baseline comparison,
and the chosen configuration under `results/random_forest_tuning_<timestamp>/`.
`models/random_forest/tuned_params.json` is the reusable selected configuration;
loading checks the dataset and preprocessing hashes. `--config` cannot be mixed
with explicit model flags. Running without `--config` still reproduces the
original baseline settings.

The current winner uses Gini, 300 trees, unlimited depth, `min_samples_split=10`,
`min_samples_leaf=1`, and `max_features="log2"`. Validation accuracy is 94.25%
and macro-F1 is 0.9422, compared with the Gini baseline's 93.10% and 0.9309.
These are selection scores on the same 174 validation rows, not independent
generalization estimates. No new test score was computed during tuning.

### RF preprocessing selection and five-fold assessment

```bash
python models/random_forest/compare_data_and_cross_validate.py
```

This workflow first compares the RF v1 preprocessing with a pinned copy of
`main` at commit `1e2acaa9959dee5557dac8182dd8f81d7990e120`, using the same
435/174 training/validation rows and the frozen tuned Gini settings. Both
versions use the identical original CSV, preserve row alignment, and produce
finite numeric features. RF v1 achieves 94.25% accuracy / 0.9422 macro-F1,
versus 91.95% / 0.9193 for main, so RF v1 remains the active data version.
Training prices missing or unparsed number 18 for RF v1 and 75 for main;
main's stricter parser deliberately rejects more prose answers. These counts
alone do not establish which parser is semantically more accurate.

The script then keeps the existing 261-row / 87-student test holdout and divides
the other 609 rows / 203 students into five groups with seed 43. This preserves
our existing holdout instead of adopting main's newer 20% holdout. Each fold
fits fresh medians and vocabularies on its training students, so feature counts
can differ between folds. The frozen Gini structural parameters are also run
with entropy; no hyperparameters are selected using these folds.

Mean validation accuracy (sample standard deviation across folds) is
92.93% (2.15 percentage points) for Gini and 93.27% (2.49 percentage points)
for entropy. Mean macro-F1 is 0.9293 and 0.9326 respectively. Results include
fold assignments, preprocessing states, fold metrics, out-of-fold predictions,
confusion matrices and source provenance under `results/random_forest_5fold_*`.
The stored RF feature CSV remains the input for the original holdout workflow;
cross-validation rebuilds features from raw responses using the same RF rules.

These folds reuse data involved in earlier selection and are not nested CV or
a fresh independent test estimate. The local holdout was evaluated earlier;
this run does not evaluate it again. The `main_reference` files are unmodified
source snapshots for reproducibility, not a merge into the shared team code.
