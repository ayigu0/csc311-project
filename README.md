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
│   ├── knn.py
│   ├── decision_tree.py
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

One file per model family, so teammates can work on separate experiments:

- `knn.py`: k-nearest neighbours.
- `decision_tree.py`: decision tree.
- `softmax_regression.py`: multiclass logistic regression.

Each file is currently a placeholder. Later, it should contain its model's
training setup, hyperparameter candidates, and tuning code. Reuse the shared
data preparation functions rather than keeping separate copies in each file.
Document model-specific preprocessing, such as feature scaling.

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

This folder is initially empty. Git tracks files, so it will appear on GitHub
once it contains a tracked output or a `.gitkeep` placeholder.

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
