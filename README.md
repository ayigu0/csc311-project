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

## Random Forest results and selected package

- [RF entry point](models/random_forest/README.md): current code, results, and reproduction commands.
- [Report-ready performance](models/random_forest/FINAL_PERFORMANCE.md): shared five-fold CV and shared local test, reported separately.
- [Selected package](models/random_forest/submit/README.md): preprocessing, model, and parameters.
- [Archived experiments](models/random_forest/archive/README.md): historical scripts, data version, tuning, and experiment records.

The current RF shared-fold mean validation accuracy is **92.10%**; its shared
174-row local-test accuracy is **89.08%** after training on all 696 development
rows. The local holdout was previously exposed in RF experiments, so this is
not an independent blind-test score.
