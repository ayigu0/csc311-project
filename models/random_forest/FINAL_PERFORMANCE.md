# Comparable model performance for the project report

Use the table below for the cross-model comparison. These are **five-fold
development cross-validation results**, not independent test-set scores.
Recorded on 2026-11-25 (America/Toronto).

## Shared evaluation protocol

All three models use `data_exploration.dataexploration.split_by_student()`:

- Original population: 290 students and 870 responses.
- Reserve 20% of students: 58 students / 174 responses, with split seed 42.
- Divide the remaining 232 students / 696 responses into five validation
  groups with fold seed 43.
- Folds 1 and 2: 185 training students / 555 rows and 47 validation students /
  141 rows. Folds 3–5: 186 training students / 558 rows and 46 validation
  students / 138 rows.
- A student's responses stay together. Train, validation, and reserved-test
  students are disjoint within each fold.
- Each development row receives exactly one out-of-fold prediction per model.
- Fit each model's preprocessing inside each training fold. Keep the selected
  model settings fixed. No test predictions are made in this evaluation.
- Report the unweighted mean and sample standard deviation across the five
  folds for both accuracy and macro-F1. Do not mix these with pooled OOF metrics
  or scores from a different holdout split.

## Report table

| Model | Mean validation accuracy | Accuracy SD | Mean validation macro-F1 | Macro-F1 SD |
|---|---:|---:|---:|---:|
| Random Forest (selected Gini) | 92.10% | 1.39 percentage points | 0.9209 | 0.0140 |
| Softmax Regression | 92.10% | 0.81 percentage points | 0.9210 | 0.0082 |
| Naive Bayes | 92.82% | 0.51 percentage points | 0.9283 | 0.0053 |

Naive Bayes has the highest point estimates in this comparison: five more
correct out-of-fold predictions than either RF or Softmax across 696 responses.
RF and Softmax have identical mean accuracy here but different per-fold errors
and macro-F1. Fold standard deviations are descriptive, not confidence intervals
or a significance test.

RF uses its selected `submit/` package: 300 Gini trees, unlimited depth,
`min_samples_split=10`, `min_samples_leaf=1`, `max_features="log2"`, seed 42.
Softmax uses its existing selected settings (learning rate 0.1, batch size 32,
L2 0.0001, seed 42). Naive Bayes uses its existing selected configuration
(minimum text frequency 2, 200 terms per field, alpha 1.0, text weight 0.75,
variance smoothing 0.01). Teammate settings and result files were not modified.

The models retain their respective preprocessing and historical tuning
procedures. Matching evaluation membership makes these scores comparable under
this protocol; it does not make the historical model-selection processes
identical or turn this assessment into nested cross-validation. RF previously
used local-holdout rows in other experiments, so the reserved partition must
not be described as never previously exposed.

## Which previous RF numbers should not enter this table?

- 94.25% was a single 174-row validation score with the older 50/20/30 split.
- 92.93% / 93.27% were Gini/entropy five-fold means over only 609 development
  rows, with 261 responses reserved. Their membership differs from this protocol.
- 88.51% was the selected Gini model's retrospective score on that older
  261-row test partition. It must not be compared with teammates' CV scores.
- The 70/80/90/95% training-size study used separate exploratory repeated
  partitions and is supplementary evidence, not this report's comparison table.

For any separate final-test table, all models need the same final training
population and the same 174-row test partition, with newly fitted preprocessing
on that training population. RF has now been evaluated this way below; a
three-model test comparison has not been run.

## RF on the shared local test partition

After freezing the selected Gini parameters, fit the model and preprocessing
on all 696 development responses (232 students), then predict the same 174
test responses (58 students) reserved by the teammates' shared split function.
This fit includes all five development folds and excludes every test student.

| Partition | Correct / total | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Full development training set | 682 / 696 | 97.99% | 0.9799 |
| Shared local test set | 155 / 174 | 89.08% | 0.8911 |

These are the RF results to use in a **separate shared-test comparison** once
the other models have also been fitted on all 696 development responses and
evaluated on these exact 174 test responses. Do not place RF's 89.08% beside
Softmax/Naive Bayes CV scores under a single test-performance heading.

The split is shared, but these rows were exposed in historical RF experiments.
This is a retrospective local-holdout evaluation, not a new independent blind
test or the course's hidden test score. No settings were adjusted in this run.

```bash
python models/random_forest/evaluate_shared_test.py
```

- [Shared-test metrics](../../results/random_forest_shared_test_20261125T160757_303151Z/metrics.csv)
- [Shared-test predictions](../../results/random_forest_shared_test_20261125T160757_303151Z/test_predictions.csv)
- [Shared-test confusion matrix](../../results/random_forest_shared_test_20261125T160757_303151Z/test_confusion_matrix.csv)
- [Shared-test protocol and settings](../../results/random_forest_shared_test_20261125T160757_303151Z/evaluation_config.json)

## Reproduction and evidence

```bash
python models/random_forest/evaluate_shared_folds.py
```

The script checks exact row and fold equality across all three models, refits
RF preprocessing separately in each fold, replays the teammates' evaluation
functions, and verifies that their existing predictions/metrics are reproduced.
It writes a new result directory without overwriting historical or teammate files.

- [Three-model comparison CSV](../../results/model_comparison_shared_folds_20261125T040231_675387Z/model_comparison.csv)
- [All 15 fold metrics](../../results/model_comparison_shared_folds_20261125T040231_675387Z/metrics_by_fold.csv)
- [Exact shared split assignments](../../results/model_comparison_shared_folds_20261125T040231_675387Z/shared_fold_assignments.csv)
- [RF out-of-fold predictions](../../results/model_comparison_shared_folds_20261125T040231_675387Z/random_forest_gini_validation_predictions.csv)
- [Protocol, model settings, and source hashes](../../results/model_comparison_shared_folds_20261125T040231_675387Z/comparison_config.json)
