"""Compare Gini and entropy with otherwise identical baseline settings.

Run from the project root: python models/random_forest/compare_random_forest.py
Only train/validation results are computed; the test split stays reserved.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.random_forest import run_baseline


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_root = PROJECT_ROOT / "results" / f"random_forest_comparison_{timestamp}"
    summaries = []
    configs = []
    predictions = []
    for criterion in ["gini", "entropy"]:
        _, metrics, run_dir = run_baseline(output_root=output_root, criterion=criterion)
        indexed = metrics.set_index("split")
        summaries.append({
            "criterion": criterion,
            "train_accuracy": indexed.loc["train", "accuracy"],
            "train_macro_f1": indexed.loc["train", "macro_f1"],
            "validation_accuracy": indexed.loc["validation", "accuracy"],
            "validation_macro_f1": indexed.loc["validation", "macro_f1"],
            "accuracy_gap": indexed.loc["train", "accuracy"] - indexed.loc["validation", "accuracy"],
            "run_directory": run_dir.name,
        })
        configs.append(json.loads((run_dir / "run_config.json").read_text()))
        predictions.append(pd.read_csv(run_dir / "validation_predictions.csv"))

    # Confirm the comparison changed only the splitting criterion.
    for key in ["data_sha256", "features", "split", "preprocessing", "versions"]:
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Comparison settings differ: {key}")
    parameters = [dict(config["hyperparameters"]) for config in configs]
    for params in parameters:
        params.pop("criterion")
    if parameters[0] != parameters[1]:
        raise ValueError("Other model hyperparameters differ.")
    pd.testing.assert_frame_equal(
        predictions[0][["source_row_index", "actual"]],
        predictions[1][["source_row_index", "actual"]],
    )

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(output_root / "comparison.csv", index=False)
    paired = predictions[0].rename(columns={"predicted": "gini_prediction"})
    paired["entropy_prediction"] = predictions[1]["predicted"]
    paired.to_csv(output_root / "paired_validation_predictions.csv", index=False)
    print("\nCriterion comparison (one fixed split and forest seed; not a stability estimate):")
    print(comparison.drop(columns="run_directory").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nComparison saved to: {output_root}")


if __name__ == "__main__":
    main()
