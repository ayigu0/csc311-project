# Selected Random Forest package

This folder isolates the selected model and preprocessing from experiments.
It can be copied outside this repository and imported with its dependencies
installed. All model-code imports stay within this folder.

| File | Purpose |
|---|---|
| `random_forest.py` | `RandomForestModel.fit()` / `.predict()` for raw survey rows |
| `data_preprocessing.py` | Selected RF v1 cleaning, encoding, and training-fitted statistics |
| `survey_schema.py` | Frozen question names and Likert mappings |
| `tuned_params.json` | Exact copy of the selected Gini configuration and its provenance |
| `preprocessing_state.json` | Historical 435-training-row medians and vocabulary snapshot |
| `package_manifest.json` | File hashes and source provenance for this isolated copy |
| `requirements.txt` | Runtime dependencies |

Selected settings: 300 trees, Gini, unlimited depth, minimum split size 10,
minimum leaf size 1, `max_features="log2"`, bootstrap sampling, random seed 42.

From the repository root:

```python
from models.random_forest.submit.random_forest import RandomForestModel

# training_rows and validation_rows are raw survey DataFrames chosen by caller.
model = RandomForestModel().fit(training_rows)
predictions = model.predict(validation_rows.drop(columns="label"))
```

When the folder is copied elsewhere and is on Python's import path, use
`from random_forest import RandomForestModel` instead. No project-root paths,
shared exploration modules, experiment outputs, or network access are needed.

`fit()` learns fresh preprocessing from the supplied training rows; it does not
reuse `preprocessing_state.json`. That JSON is the historical state snapshot
for reproducing the original selected holdout model's feature representation.
The caller controls which rows are used for training. This package imposes no
new train/validation split and makes no new hyperparameter choice.

The parent RF folder retains experiments, tests, historical data and the English
experiment log; the repository's `results/` retains all previous outputs.
This is the selected **development source package**, not a trained weights file
or the final course `pred.py`. Its sklearn prediction code still needs the
allowed-library inference/export step before that final submission.
