"""Create the versioned RF feature CSV; fit all learned rules on training rows.

Run: python models/random_forest/data_preprocessing.py
For unseen responses: transform_features(raw_frame, saved_metadata['state']).
The CSV is bound to the saved student split. Refit preprocessing within each
training fold if cross-validation is introduced; do not cross-validate this CSV.
"""

import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_exploration.dataexploration import (
    DATA_FILE, LIKERT_COLS, LIKERT_MAP, rename_features, split_by_student,
)

VERSION = "rf_preprocessed_v1"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
FEATURE_FILE = OUTPUT_DIR / "rf_features_v1.csv"
METADATA_FILE = OUTPUT_DIR / "rf_preprocessing_v1.json"
SPLIT_SEED, VAL_FRAC, TEST_FRAC = 42, 0.2, 0.3
LABELS = ["Soccer", "Baseball", "Basketball"]
MULTI_SELECT = {
    "play_locations": ["Indoor facility", "Outdoor public space", "School or university facility",
                       "Community recreation facility", "Private sports facility", "At home or nearby", "Others"],
    "play_seasons": ["Winter", "Spring", "Summer", "Fall"],
    "music_genres": ["Pop", "Rock", "Hip-hop or rap", "Electronic or dance", "Country",
                     "Latin", "Classical or jazz", "Folk or traditional", "Other", "No particular type"],
}
TEXT_COLS = ["food_and_drinks", "associated_colour", "associated_song", "watching_feelings"]
ANSWER_COLS = ["beginner_ease", *LIKERT_COLS, *MULTI_SELECT, *TEXT_COLS, "ticket_price"]
# Exact normalized matches only: do not erase phrases such as "nothing but fun".
# Bare "none" remains meaningful in association fields, but becomes zero for price.
UNKNOWN = {"", "n/a", "na", "nan", "null", "-", ".", "?", "idk", "i don't know", "not sure", "unknown"}
STOP_WORDS = set("a an the and or of to in on for with at by i me my it its is are was this that sport sports".split())
TEXT_MIN_DF, TEXT_MAX_FEATURES = 2, 80
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*k?(?!\w|\.\d)")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = value.translate(str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", value).strip()


def is_unknown(text):
    return text in UNKNOWN or text.rstrip(".!?") in UNKNOWN


def clean_answers(raw):
    data = rename_features(raw)
    missing = set(ANSWER_COLS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing survey columns: {sorted(missing)}")
    cleaned = data[ANSWER_COLS].apply(lambda col: col.map(normalize_text))
    for col in ANSWER_COLS:
        cleaned[col] = cleaned[col].map(lambda text: "" if is_unknown(text) else text)
    return cleaned


def parse_ticket_price(value):
    """Return (CAD estimate, is_range); ambiguous/non-CAD values become NaN.

    Bare numbers/$ are assumed CAD. A single range uses its midpoint, even if
    bounds are reversed. Multiple scenarios, ratings, and foreign currencies
    are not guessed. Large nonnegative prices are retained, without clipping.
    """
    text = normalize_text(value)
    if is_unknown(text):
        return np.nan, 0
    if re.search(r"[€£¥₹]|\b(?:usd|eur|gbp|rmb|cny|inr|us dollars?|euros?|pounds?)\b|\bus\s*\$", text):
        return np.nan, 0
    if re.search(r"\bscale\b|%", text):
        return np.nan, 0
    # Normalize thousands separators before finding amounts.
    text = re.sub(r"(?<=\d)[, ](?=\d{3}(?:\D|$))", "", text)
    # Split currency suffixes such as 100CAD/200dollars without changing digits.
    text = re.sub(r"(?<=\d)(?=cad|dollars?)", " ", text)
    if re.search(r"(?:^|\$)\s*-\s*\d", text):
        return np.nan, 0
    matches = list(NUMBER.finditer(text))
    amounts = [float(m.group().strip().rstrip("k").strip()) *
               (1000 if m.group().strip().endswith("k") else 1) for m in matches]
    if len(amounts) == 1:
        monetary = bool(re.search(r"\$|\bcad\b|dollars?|\bpay\b|\bspend\b", text))
        bare = bool(re.fullmatch(r"(?:around |about |approximately )?\d+(?:\.\d+)?\s*k?\.?", text))
        return (amounts[0], 0) if monetary or bare else (np.nan, 0)
    if len(amounts) == 2:
        between = text[matches[0].end():matches[1].start()]
        between = re.sub(r"\$|\bcad\b|dollars?", "", between).strip()
        if between in {"-", "to"}:
            return sum(amounts) / 2, 1
        return np.nan, 0
    if amounts:
        return np.nan, 0
    if text.rstrip(".") in {"none", "nothing", "zero", "unwilling", "not willing", "no willing", "not really willing to"}:
        return 0.0, 0
    if re.search(r"\b(?:would not|wouldn't|wouldnt|won't|wont) pay\b|you need to pay me", text):
        return 0.0, 0
    # Conservative support for written amounts below 100, e.g. forty-five dollars.
    words = re.sub(r"\bdollars?\b|\bcad\b|[.-]", " ", text).split()
    units = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
    tens = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
    if len(words) == 1 and words[0] in units | tens:
        return float((units | tens)[words[0]]), 0
    if len(words) == 2 and words[0] in tens and words[1] in units and 0 < units[words[1]] < 10:
        return float(tens[words[0]] + units[words[1]]), 0
    return np.nan, 0


def tokens(text):
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return set(re.findall(r"\b[^\W\d_]{2,}\b", text)) - STOP_WORDS


def base_features(cleaned):
    features = pd.DataFrame(index=cleaned.index)
    ease = pd.to_numeric(cleaned["beginner_ease"], errors="coerce")
    features["beginner_ease"] = ease.where(ease.between(0, 10))
    mapping = {key.casefold(): value for key, value in LIKERT_MAP.items()}
    for col in LIKERT_COLS:
        features[col] = cleaned[col].map(mapping)
    prices = cleaned["ticket_price"].map(parse_ticket_price)
    features["ticket_price_cad"] = prices.map(lambda value: value[0])
    features["ticket_price_is_range"] = prices.map(lambda value: value[1])
    for col in ["beginner_ease", *LIKERT_COLS, "ticket_price_cad"]:
        features[f"{col}__missing"] = features[col].isna().astype(int)
    for col, categories in MULTI_SELECT.items():
        answers = cleaned[col].map(lambda text: {part.strip() for part in text.split(",") if part.strip()})
        known = {category.casefold() for category in categories}
        features[f"{col}__missing"] = cleaned[col].eq("").astype(int)
        features[f"{col}__unknown_option"] = answers.map(lambda selected: int(bool(selected - known)))
        for category in categories:
            suffix = re.sub(r"\W+", "_", category.casefold()).strip("_")
            features[f"{col}__{suffix}"] = answers.map(lambda selected: int(category.casefold() in selected))
    for col in TEXT_COLS:
        features[f"{col}__missing"] = cleaned[col].eq("").astype(int)
    features["all_answers_missing"] = cleaned.eq("").all(axis=1).astype(int)
    return features


def fit_preprocessor(train_raw):
    """Learn medians and document-frequency vocabularies, without reading labels."""
    cleaned = clean_answers(train_raw)
    base = base_features(cleaned)
    vocabularies = {}
    for col in TEXT_COLS:
        counts = {}
        for text in cleaned[col]:
            for token in tokens(text):
                counts[token] = counts.get(token, 0) + 1
        vocabularies[col] = sorted(
            (token for token, count in counts.items() if count >= TEXT_MIN_DF),
            key=lambda token: (-counts[token], token),
        )[:TEXT_MAX_FEATURES]
    feature_columns = list(base.columns) + [
        f"{col}__word_{token}" for col in TEXT_COLS for token in vocabularies[col]
    ]
    return {
        "version": VERSION,
        "medians": base.median().fillna(0).to_dict(),
        "vocabularies": vocabularies,
        "feature_columns": feature_columns,
        "text_min_df": TEXT_MIN_DF,
        "text_max_features_per_field": TEXT_MAX_FEATURES,
        "training_rows": len(train_raw),
        "training_missing_before_imputation": base.isna().sum().to_dict(),
    }


def transform_features(raw, state):
    """Apply frozen rules, preserving row order/index; label and ID are optional."""
    cleaned = clean_answers(raw)
    base = base_features(cleaned).fillna(state["medians"])
    text_features = {}
    for col in TEXT_COLS:
        word_sets = cleaned[col].map(tokens)
        for token in state["vocabularies"][col]:
            text_features[f"{col}__word_{token}"] = word_sets.map(lambda words: int(token in words))
    features = pd.concat([base, pd.DataFrame(text_features, index=raw.index)], axis=1)
    features = features[state["feature_columns"]].astype(float)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Preprocessing produced non-finite feature values.")
    return features


def generate_dataset():
    raw = pd.read_csv(DATA_FILE)
    if raw["label"].isna().any() or not raw["label"].isin(LABELS).all():
        raise ValueError("Invalid or missing class labels.")
    if raw.duplicated(["id", "label"]).any():
        raise ValueError("Duplicate student/class rows require review.")
    train, val, test = split_by_student(raw["id"], VAL_FRAC, TEST_FRAC, SPLIT_SEED)
    state = fit_preprocessor(raw.loc[train])
    # Held-out rows are transformed with frozen rules; never fit or tune on them.
    features = transform_features(raw, state)
    output = raw[["id", "label"]].copy()
    output.insert(0, "source_row_index", raw.index)
    output["split"] = np.select([train, val, test], ["train", "validation", "test"], default="")
    output = pd.concat([output, features], axis=1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(FEATURE_FILE, index=False)
    metadata = {
        "version": VERSION,
        "source_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(DATA_FILE),
        "preprocessing_code_sha256": sha256(__file__),
        "feature_file_sha256": sha256(FEATURE_FILE),
        "split": {"seed": SPLIT_SEED, "val_fraction": VAL_FRAC, "test_fraction": TEST_FRAC,
                  "rows": {"train": int(train.sum()), "validation": int(val.sum()), "test_reserved": int(test.sum())}},
        "policies": {
            "row_policy": "retain every row, including empty answers; include missing indicators",
            "numeric_imputation": "training median; all-missing training columns use zero",
            "multi_select": "fixed survey categories; absent/unknown answers have separate indicators",
            "text": "normalized binary words; vocabulary fitted on training rows only; exact unknown markers removed; bare none retained",
            "ticket_price": "assume CAD for bare numbers/$; midpoint for one range; foreign currencies and ambiguous answers become missing; keep nonnegative outliers",
            "scaling": "none",
            "cross_validation": "refit preprocessing within each training fold, starting from raw data",
        },
        "state": state,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(output)} rows, {features.shape[1]} numeric features: {FEATURE_FILE}")
    print(f"Frozen preprocessing rules: {METADATA_FILE}")
    return output, metadata


def load_preprocessed_dataset():
    if not FEATURE_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Run python models/random_forest/data_preprocessing.py first.")
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if (metadata["version"] != VERSION or metadata["source_sha256"] != sha256(DATA_FILE)
            or metadata["preprocessing_code_sha256"] != sha256(__file__)
            or metadata["feature_file_sha256"] != sha256(FEATURE_FILE)):
        raise ValueError("RF data is stale or modified; rerun data_preprocessing.py.")
    data = pd.read_csv(FEATURE_FILE)
    columns = metadata["state"]["feature_columns"]
    if list(data.columns) != ["source_row_index", "id", "label", "split", *columns]:
        raise ValueError("Unexpected RF dataset schema.")
    if not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError("RF dataset contains invalid numeric features.")
    return data.set_index("source_row_index"), metadata


if __name__ == "__main__":
    generate_dataset()
