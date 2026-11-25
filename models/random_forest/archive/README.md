# Historical RF experiments

This archive preserves baseline Gini/entropy comparisons, tuning, preprocessing
selection, old 50/20/30 splits, and the 70/80/90/95% training-size study.
Use [the current performance report](../FINAL_PERFORMANCE.md) for reporting.

- [Experiment log](EXPERIMENT_LOG.md): historical decisions and measurements.
- [Historical workflow](WORKFLOW.md): commands from before this reorganization.
- [Historical result directories](../../../results/random_forest/archive/).
- [Relocation manifest](manifest.json): original paths, new paths, and SHA-256 hashes.
- `data/`: frozen features and fitted preprocessing for the old split; current
  evaluation refits the selected preprocessing on each training population.

Archived scripts are frozen source records. Their imports and relative paths
assume the original layout. Reproduce an old run in a separate checkout:

```bash
git worktree add --detach ../csc311-rf-history f62c0e6135d7ab87417543287e766b86895c186d
```

All original source and result bytes are retained; only this log's Markdown
links were updated. Embedded historical paths and hashes remain unchanged.
