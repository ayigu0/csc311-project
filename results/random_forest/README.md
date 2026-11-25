# RF result index

- [Current shared CV](current/shared_cv/model_comparison.csv): the three-model,
  five-fold development comparison; 696 responses and 232 students.
- [Current shared test](current/shared_test/metrics.csv): selected RF fitted on
  696 development responses and evaluated on 174 local-test responses.
- [Performance report](../../models/random_forest/FINAL_PERFORMANCE.md): interpretation,
  protocol, and links to predictions and provenance.
- [Archive](archive/): all older experiment outputs, preserved byte-for-byte.
- `runs/`: timestamped outputs created by subsequent evaluations. Reviewed
  `current/` artifacts are never overwritten automatically.

The current directories contain relocated original run artifacts. Their JSON
provenance retains original paths and hashes; see the
[relocation manifest](../../models/random_forest/archive/manifest.json).
The reused local holdout is not a fresh blind test.
