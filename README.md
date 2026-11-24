# CSC311 Project

Workspace for our four-person team's data exploration and model experiments.

## Structure

```text
Project/
├── data/
│   └── csc311_project_training_202609.csv
├── data_exploration/
│   └── dataexploration.py
├── models/
│   ├── naive_bayes.py
│   ├── random_forest/
│   │   ├── data_preprocessing.py
│   │   ├── random_forest.py
│   │   ├── compare_random_forest.py
│   │   └── data/
│   └── softmax_regression.py
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
- `split_by_student(ids, ...)`: return training, validation, and test masks
  while keeping each student's responses together.

Running the file currently only loads the CSV and renames its columns. The
other functions are available to call when needed. The feature builder still
needs extending to cover the other survey questions.

### `models/`

Separate model workspaces let teammates work on their own experiments:

- `naive_bayes.py`: Naive Bayes placeholder.
- `random_forest/`: RF preprocessing, training, comparison, and versioned data.
- `softmax_regression.py`: multiclass logistic regression.

RF has a runnable development workflow. The other model files remain
placeholders. Each model should document its preprocessing and keep the
shared student-level split for comparable evaluation.

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

RF experiment folders preserve the input version, settings, and validation
results for each run, including the earlier eight-feature baselines.

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
Python Console** to keep `df` available for interactive commands, such as
`df.columns`.

CSV paths are resolved relative to the Python files, so loading also works
when your working directory is different from the project folder.

## Shared experiment conventions

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

### Team conventions

- Use the same original dataset and student-level split for every model.
  The current splitting defaults are seed `42`, validation fraction `0.2`,
  and test fraction `0.3`; the remaining students are used for training.
  These defaults take effect when `split_by_student()` is called.
- Split before exploration or dropping rows. Use the training split for
  exploration, and keep student IDs out of the model's predictors.
- Learn preprocessing values, including imputation values, scaling statistics,
  and text vocabularies, from the training data only. During cross-validation,
  learn them separately within each training fold.
- Tune every model family using validation data and compare accuracy and
  macro-F1 on the same validation split. Reserve the local test set for final
  evaluation after model and preprocessing choices are fixed.
- The supplied baseline evaluates the test set each time it runs. Use the
  team's validation workflow for ongoing model selection.
- Work on task branches and have another teammate review changes before
  merging. Include the experiment settings with any results you share.
