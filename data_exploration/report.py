"""Generate the exploration artifacts from the first development training fold."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .dataexploration import (
    EASE_COL, LIKERT_COLS, LIKERT_MAP, load_dataset, rename_features, split_by_student,
)
from .preprocessing import (
    FIELDS, NUMERIC, PRICE, TEXT, SurveyPreprocessor, normalize_answers,
    numeric_answers, selections,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def data_quality(data, answers, numeric, features):
    raw = data.loc[:, list(FIELDS)].astype("string").apply(lambda values: values.str.strip())
    blanks = raw.isna() | raw.eq("").fillna(False)
    markers = answers.isna() & ~blanks
    invalid = pd.DataFrame(False, index=data.index, columns=FIELDS)
    invalid[list(NUMERIC)] = answers[list(NUMERIC)].notna() & numeric.isna()
    unavailable = blanks | markers | invalid
    return pd.DataFrame({
        "answered_rows": (~unavailable).sum(),
        "raw_blanks": blanks.sum(),
        "missing_markers": markers.sum(),
        "invalid_answers": invalid.sum(),
        "missing_pct": unavailable.mean().mul(100).round(2),
        "distinct_normalized_answers": answers.mask(invalid).nunique(),
        "feature_columns": [
            sum(name == field or name.startswith(f"{field}__") for name in features)
            for field in FIELDS
        ],
    }).rename_axis("field")


def summarize_prices(answers, prices, labels):
    grouped = prices.groupby(labels)
    summary = grouped.agg(rows="size", parsed="count", median_cad="median",
                          minimum_cad="min", maximum_cad="max")
    summary["missing_answers"] = answers.isna().groupby(labels).sum()
    summary["parse_failures"] = (prices.isna() & answers.notna()).groupby(labels).sum()
    summary["parsed_pct"] = (100 * summary["parsed"] / summary["rows"]).round(2)
    summary["q1_cad"] = grouped.quantile(0.25)
    summary["q3_cad"] = grouped.quantile(0.75)
    q1, q3 = prices.quantile([0.25, 0.75])
    iqr = q3 - q1
    outside = prices.lt(q1 - 1.5 * iqr) | prices.gt(q3 + 1.5 * iqr)
    summary["outliers_global_iqr"] = outside.groupby(labels).sum()
    return summary


def report_multiselect(answers, labels, options_by_field, output):
    rows = []
    for field, options in options_by_field.items():
        field_answers = answers[field].dropna()
        chosen = selections(field_answers)
        for option in options:
            selected = chosen.map(lambda values: option in values)
            for sport, values in selected.groupby(labels.loc[selected.index]):
                rows.append((field, option, sport, values.sum(), len(values), 100 * values.mean()))
    rates = pd.DataFrame(rows, columns=["field", "option", "sport", "selected", "answered", "percent"])
    rates.to_csv(output / "exploration_multiselect_rates.csv", index=False, float_format="%.2f")
    matrix = rates.pivot(index=["field", "option"], columns="sport", values="percent")
    fig, ax = plt.subplots(figsize=(8, 9))
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix)), [f"{field.replace('_', ' ')} · {option}" for field, option in matrix.index])
    for row in range(len(matrix)):
        for col in range(len(matrix.columns)):
            value = matrix.iloc[row, col]
            ax.text(col, row, f"{value:.0f}%", ha="center", va="center",
                    color="white" if value >= 60 else "#14273b")
    fig.colorbar(image, ax=ax, label="Percent of respondents who answered")
    ax.set_title("Survey options selected by sport")
    fig.savefig(output / "exploration_multiselect_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return rates


def report_feature_summary(answers, numeric, quality, multiselect_rates, output):
    """Write the report's Table 1.3 using the first training fold."""
    fields = (
        (EASE_COL, "Beginner ease", "Numeric"),
        ("strategy_importance", "Strategy", "Ordinal"),
        ("injury_risk", "Injury risk", "Ordinal"),
        ("individual_brilliance", "Individual brilliance", "Ordinal"),
        ("play_locations", "Places to play", "Multi-select"),
        ("play_seasons", "Seasons", "Multi-select"),
        ("music_genres", "Music genres", "Multi-select"),
        ("food_and_drinks", "Food and drinks", "Text"),
        ("associated_colour", "Colour", "Text"),
        ("associated_song", "Song", "Text"),
        ("watching_feelings", "Watching feelings", "Text"),
        (PRICE, "Ticket price", "Amount/text"),
    )

    # The report counts blanks and missing markers before attempting to parse prices.
    missing = answers.isna().mean().mul(100)
    ease = numeric[EASE_COL].dropna()
    prices = numeric[PRICE].dropna()
    distributions = {
        EASE_COL: f"{ease.min():g}-{ease.max():g}; median {ease.median():g}" if len(ease)
        else "No valid ratings",
        PRICE: f"{len(prices)}/{len(answers)} parsed; max {prices.max():g}" if len(prices)
        else f"0/{len(answers)} parsed",
    }
    rating_labels = {score: label for label, score in LIKERT_MAP.items()}
    for field in LIKERT_COLS:
        modes = numeric[field].mode()
        distributions[field] = f"Mode: {rating_labels[modes.iloc[0]]}" if len(modes) else "No valid ratings"
    for field, rates in multiselect_rates.groupby("field"):
        counts = rates.groupby("option")["selected"].sum()
        distributions[field] = f"{len(counts)} options; top: {counts.idxmax().capitalize()}"
    for field in TEXT:
        distinct = quality.at[field, "distinct_normalized_answers"]
        distributions[field] = f"{distinct} distinct answers"

    table = pd.DataFrame(
        [(name, kind, missing[field], distributions.get(field, "0 options")) for field, name, kind in fields],
        columns=["Feature", "Type", "Missing (%)", "Distribution / issue"],
    )
    table.to_csv(output / "exploration_table_1_3.csv", index=False, float_format="%.2f")


def report_text(answers, labels, output):
    rows = []
    for field in TEXT:
        for sport, group in answers[field].dropna().groupby(labels):
            for response, count in group.value_counts().head(5).items():
                rows.append((field, sport, response, count, len(group), 100 * count / len(group)))
    pd.DataFrame(rows, columns=["field", "sport", "response", "count", "answered", "percent"]).to_csv(
        output / "exploration_text_associations.csv", index=False, float_format="%.2f"
    )


def report_ratings(numeric, labels, output):
    ratings = numeric[[EASE_COL, *LIKERT_COLS]]
    ratings.groupby(labels).agg(["count", "median", "mean"]).round(2).to_csv(
        output / "exploration_ratings_summary.csv"
    )
    ratings[LIKERT_COLS].corr(method="spearman").round(3).to_csv(
        output / "exploration_likert_correlations.csv"
    )
    titles = ("Beginner ease", "Strategy", "Injury risk", "Individual brilliance")
    sport_colours = {"Baseball": "#4477aa", "Basketball": "#ee7733", "Soccer": "#228833"}
    sports = sorted(labels.unique())
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 4.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1)
    for field, title, ax in zip(ratings, titles, axes.flat):
        samples = [ratings.loc[labels.eq(sport), field].dropna() for sport in sports]
        artists = ax.boxplot(
            samples, widths=0.55, patch_artist=True,
            medianprops={"color": "#a00050", "linewidth": 2.5},
        )
        for sport, box in zip(sports, artists["boxes"]):
            box.set(facecolor=sport_colours.get(sport, "#999999"), alpha=0.55)

        ax.set_xticks(range(1, len(sports) + 1), sports)
        ax.set_title(title, fontsize=11)
        if field == EASE_COL:
            ax.set_ylabel("Rating (0-10)")
            ax.set_yticks([0, 2.5, 5, 7.5, 10])
            ax.yaxis.set_major_formatter("{x:.1f}")
            ax.set_ylim(-0.5, 10.5)
        else:
            ax.set_ylabel("Agreement (1-5)")
            ax.set_yticks(range(1, 6))
            ax.set_ylim(0.5, 5.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(output / "exploration_ratings_by_sport.png", dpi=220)
    plt.close(fig)


def main():
    data = rename_features(load_dataset())
    _, folds = split_by_student(data["id"])
    train = data.loc[folds[0][0]]
    labels = train["label"]
    answers = normalize_answers(train)
    numeric = numeric_answers(answers)
    preprocessor = SurveyPreprocessor().fit(train)
    features = preprocessor.transform(train)
    RESULTS.mkdir(parents=True, exist_ok=True)

    quality = data_quality(train, answers, numeric, features)
    quality.to_csv(RESULTS / "exploration_missing_values.csv")
    summarize_prices(answers[PRICE], numeric[PRICE], labels).to_csv(RESULTS / "exploration_ticket_prices.csv")
    multiselect_rates = report_multiselect(answers, labels, preprocessor.options_, RESULTS)
    report_feature_summary(answers, numeric, quality, multiselect_rates, RESULTS)
    report_text(answers, labels, RESULTS)
    report_ratings(numeric, labels, RESULTS)
    overview = labels.value_counts().rename("responses").to_frame()
    overview["students"] = train.groupby("label")["id"].nunique()
    overview["all_answers_missing"] = answers.isna().all(axis=1).groupby(labels).sum()
    overview.to_csv(RESULTS / "exploration_class_balance.csv")
    print(f"Explored {len(train)} responses from {train['id'].nunique()} students.")
    print(f"Encoded features: {features.shape[1]}. Outputs: {RESULTS}")


if __name__ == "__main__":
    main()
