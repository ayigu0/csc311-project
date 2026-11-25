# Random Forest

Start with [FINAL_PERFORMANCE.md](FINAL_PERFORMANCE.md) for the report-ready
results and the distinction between development CV and local test scores.

| Location | Purpose |
|---|---|
| [submit/](submit/README.md) | Selected preprocessing, Gini model, and frozen parameters |
| [Shared CV results](../../results/random_forest/current/shared_cv/) | Three models on the same five development folds |
| [Shared test results](../../results/random_forest/current/shared_test/) | Selected RF trained on 696 rows, evaluated on 174 shared test rows |
| [archive/](archive/README.md) | Old scripts, feature version, tuning records, and experiment log |
| [Archived results](../../results/random_forest/archive/) | Historical experiment outputs |

The selected RF has 300 Gini trees, unlimited depth, `min_samples_split=10`,
`min_samples_leaf=1`, `max_features="log2"`, and seed 42. Fit preprocessing only
on the training population. Do not use the old frozen feature CSV for shared CV.

Run from the repository root:

```bash
python models/random_forest/evaluate_shared_folds.py
python models/random_forest/evaluate_shared_test.py
python -m unittest models.random_forest.test_data_preprocessing models.random_forest.test_submit_package
```

New runs go to timestamped directories under `results/random_forest/runs/`.
They do not replace the reviewed `current/` results. Historical provenance paths
in result JSON files refer to the original run layout; use the archive manifest
to locate relocated files. The selected `submit/` package is unchanged and is
not yet the course's restricted-library final prediction bundle.
