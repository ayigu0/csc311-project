# CSC311 Project

Workspace for our four-person team's data exploration and model experiments.

## Structure

```text
Project/
├── data/
│   └── csc311_project_training_202609.csv
├── data_exploration/
│   ├── dataexploration.py
│   ├── preprocessing.py
│   └── report.py
├── models/
│   ├── naive_bayes/
│   ├── random_forest/
│   │   └── submit/
│   └── softmax/
├── results/
├── project_baseline.py
├── pred_example.py
├── README.md
├── requirements.txt
└── .gitignore
```

## What belongs in each folder

### `data/`

The original course-provided CSV. Keep the original responses unchanged so
everyone starts from the same data. Perform cleaning and transformations in
code rather than editing the CSV manually.

### `data_exploration/`

Code for understanding the survey data: feature types and distributions,
class balance, missing answers, inconsistencies, and relationships with the
label. Exploration notebooks can also go here if we use them later.

The current `dataexploration.py` provides:

- `load_dataset()`: read the CSV into a pandas DataFrame.
- `rename_features(df)`: replace long questions with short column names.
- `make_features(df)`: convert the baseline's subset of ease, Likert, and
  season answers into numeric features, retaining missing values.
- `split_by_student(ids, ...)`: return the held-out test mask and five
  (training, validation) mask pairs for grouped development folds.

Running the file prints student and response counts for the holdout and each
fold. Use `SurveyPreprocessor` in `data_exploration/preprocessing.py` for the
full set of 12 questions. The original `make_features` helper covers the
baseline questions.

Generate the exploration tables and plots from the first training fold:

```bash
python -m data_exploration.report
```

The outputs include class counts, raw blanks and recognized missing markers,
invalid ratings and price parse failures, rating distributions and Spearman
correlations, ticket-price summaries, multi-select rates and their heatmap,
and frequent normalized text answers by sport.

### Shared survey preprocessing

```python
from data_exploration.dataexploration import load_dataset, split_by_student
from data_exploration.preprocessing import SurveyPreprocessor

data = load_dataset()
test, folds = split_by_student(data["id"])
train, validation = folds[0]
preprocessor = SurveyPreprocessor().fit(data.loc[train])
X_train = preprocessor.transform(data.loc[train], model="softmax")
X_validation = preprocessor.transform(data.loc[validation], model="softmax")
y_train = data.loc[train, "label"]
```

Fit a new instance in each fold. For final evaluation, fit on the full
development set and transform the held-out test set. The same fitted instance
can transform new survey responses, including partially filled responses.
Student IDs and labels are excluded. Response rows, including entirely blank
ones, are retained and missing answers are flagged.

- Ratings are validated against their scales. Ease and price use training
  medians, and agreement levels use training modes. If an entire numeric
  field is unavailable in training, the fallback is zero for ease and price
  and neutral for agreement levels.
- Missing markers are matched against the whole normalized answer. Explicit
  no-association answers are retained in free-text fields. Ticket-price parsing
  accepts clear CAD amounts, range midpoints, common written numbers, and
  explicit zero-payment answers. Unclear answers and foreign currencies are
  unavailable. Unmarked amounts are assumed to be CAD.
- Multi-select choices become binary columns. Text vocabularies are learned
  separately for each field using a minimum training frequency of two responses
  and a maximum of 200 terms per field. Unseen choices and words add no columns.
- Use `model="random_forest"` for unscaled ratings and binary text,
  `model="softmax"` for standardized numeric values and TF-IDF text, or
  `model="naive_bayes"` for unscaled ratings and binary text. The Naive Bayes
  trainer can use `LIKERT_COLS` for categorical likelihoods, ease and price
  for Gaussian likelihoods, and the indicator columns for Bernoulli likelihoods.

### `models/`

The Softmax and Naive Bayes folders contain their training, tuning and
evaluation workflows using the shared preprocessing interface. The Random
Forest folder contains its selected preprocessing, model, and experiment history.
Its isolated selected source package is in [models/random_forest/submit](models/random_forest/submit/README.md).

### `results/`

Generated outputs from exploration and model experiments: plots, summary
tables, confusion matrices, and experiment records. Keep the code that
generates them in `data_exploration/` or `models/`.

Example output names to use later:

- `exploration_missing_values.csv`
- `exploration_feature_distributions.png`
- `knn_results.csv`
- `decision_tree_results.csv`
- `softmax_results.csv`

Record the model, feature set, hyperparameters, split settings, validation
accuracy, and validation macro-F1 alongside each experiment's results. Use
distinct filenames when comparing runs so teammates' results are preserved.

The exploration outputs can be regenerated with the command above.

## Files in the project root

- `project_baseline.py`: the supplied baseline workflow, with its CSV path
  updated for this layout. Running it trains kNN and reports test accuracy.
- `pred_example.py`: the course example of the required prediction interface.
- `requirements.txt`: the shared Python dependencies; update it when adding
  a dependency needed by the team.
- `.gitignore`: excludes local IDE settings, virtual environments, and Python
  cache files from Git.

## Setup

Open this folder in PyCharm and select your project Python interpreter
(Python 3.10 or later). Install the dependencies in that environment:

```bash
python -m pip install -r requirements.txt
```

## Load the dataset

From the `Project` folder, run:

```bash
python data_exploration/dataexploration.py
```

In PyCharm, open `data_exploration/dataexploration.py` and use **Run File in
Python Console** to keep `data` available for interactive commands, such as
`data.columns`.

CSV paths are resolved relative to the Python files, so loading also works
when your working directory is different from the project folder.

## Shared experiment conventions

- Use the same original dataset and shared student-level split for every model.
  Hold out 20% of students with seed `42` and split the remaining students into
  five development folds with seed `43`.
- Create the split before exploration or dropping rows. For exploration, use
  the first fold's training mask (`folds[0][0]`). Keep student IDs out of the
  model's predictors.
- Fit imputation values, scaling statistics, and text vocabularies within each
  training fold.
- Compare models on the same five validation folds using accuracy and macro-F1.
  Reserve the held-out test set for final evaluation.
- The supplied baseline uses its original split as a standalone course example.
- Work on task branches and have another teammate review changes before
  merging. Include the experiment settings with any results you share.

## Selected RF package and preserved experiments

`models/random_forest/submit/` contains the selected model source, RF v1 cleaning
code, frozen survey schema, selected hyperparameters and historical preprocessing
state. Its imports do not depend on the parent experiments or shared exploration
code. See the [package README](models/random_forest/submit/README.md) for usage
and the remaining distinction from the course's final restricted-library pred.py.

Experiments and tests remain in `models/random_forest/`; generated results remain
in `results/`. The [English experiment log](models/random_forest/EXPERIMENT_LOG.md)
records all prior comparisons and the training-size experiment.

RF historical runs retain their 50/20/30 split through a frozen RF-specific
`survey_schema.py`; shared Softmax/Naive Bayes code retains main's newer 20%
holdout and development-fold API. These historical scores use different
partitions and must not be treated as a matched three-model comparison. The
explicit `preprocessing_compatibility.json` mapping allows the verified schema
import migration while preserving original RF data, parameter and result hashes.

Verify the integration and isolated package with:

```bash
python -m unittest models.random_forest.test_data_preprocessing models.random_forest.test_submit_package -v
```

### Random Forest preprocessing and training

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
