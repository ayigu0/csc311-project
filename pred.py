"""Predict survey sports using the saved full-data Naive Bayes model."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# These cleaning rules mirror the training survey preprocessor.
EASE_COL = "beginner_ease"
LIKERT_COLS = ["strategy_importance", "injury_risk", "individual_brilliance"]
LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
MULTI = ("play_locations", "play_seasons", "music_genres")
TEXT = ("food_and_drinks", "associated_colour", "associated_song", "watching_feelings")
PRICE = "ticket_price"
NUMERIC = (EASE_COL, *LIKERT_COLS, PRICE)
FIELDS = (*NUMERIC, *MULTI, *TEXT)
MARKERS = {"", "-", "—", "n/a", "na", "n.a.", "unknown", "not applicable",
           "i don't know", "i do not know", "idk"}
NO_ASSOCIATION = {"none", "no association", "nothing"}
LIKERT = {key.casefold(): value for key, value in LIKERT_MAP.items()}
ONES = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(), range(20)))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
AMOUNT = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"
CAD = r"cad|c\$|ca\$|\$|canadian dollars?|dollars?"
TOKEN_PATTERN = r"(?u)\b\w+\b"


def normalize_answers(frame, column_names):
    """Standardize raw or renamed survey columns, preserving substantive answers."""
    answers = frame.rename(columns=column_names).reindex(columns=FIELDS).astype("string")
    for field in FIELDS:
        value = answers[field].str.casefold().str.strip().str.replace(r"\s+", " ", regex=True)
        markers = MARKERS if field in TEXT else MARKERS | NO_ASSOCIATION
        if field == PRICE:
            markers = markers - {"nothing"}
        value = value.mask(value.isin(markers))
        if field in TEXT:
            value = value.replace(dict.fromkeys(NO_ASSOCIATION, "none"))
        if field == "food_and_drinks":
            value = value.str.replace(r"\bhotdogs?\b", "hot dog", regex=True)
        elif field == "associated_colour":
            value = value.replace({"grey": "gray"})
        elif field == "associated_song":
            value = value.str.replace(
                r"^waka waka(?: \(this time for africa\))?"
                r"(?:(?: by| [-–—]) shakira(?: \(2010\))?)?\.?$",
                "waka waka", regex=True,
            )
        answers[field] = value
    return answers


def _under_hundred(words):
    if len(words) == 1:
        return ONES.get(words[0], TENS.get(words[0]))
    if len(words) == 2 and words[0] in TENS and 0 < ONES.get(words[1], 0) < 10:
        return TENS[words[0]] + ONES[words[1]]
    return None


def _under_thousand(words):
    if len(words) >= 2 and words[1] == "hundred" and 0 < ONES.get(words[0], 0) < 10:
        rest = words[2:]
        if rest and rest[0] == "and":
            rest = rest[1:]
            if not rest:
                return None
        tail = _under_hundred(rest) if rest else 0
        return None if tail is None else ONES[words[0]] * 100 + tail
    return _under_hundred(words)


def _amount(value):
    value = re.sub(rf"^(?:(?:{CAD})\s*)+|(?:\s*(?:{CAD}))+$", "", value).strip()
    if re.fullmatch(AMOUNT, value):
        amount = float(value.replace(",", ""))
        return amount if np.isfinite(amount) else None
    words = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", value).split()
    if words.count("thousand") == 1:
        split = words.index("thousand")
        head = _under_thousand(words[:split])
        rest = words[split + 1:]
        if rest and rest[0] == "and":
            rest = rest[1:]
            if not rest:
                return None
        tail = _under_thousand(rest) if rest else 0
        return head * 1000 + tail if head and tail is not None else None
    return _under_thousand(words)


def parse_ticket_price(value):
    """Return a CAD amount, a range midpoint, or NaN for unavailable prices."""
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).strip().casefold()).rstrip(".!?")
    value = value.replace("’", "'")
    if value in {"free", "nothing", "i wouldn't pay", "i would not pay",
                 "wouldn't pay", "would not pay", "i wouldn't pay anything"}:
        return 0.0
    amount = _amount(value)
    if amount is not None:
        return float(amount)
    value = re.sub(r"^between\s+(.+)\s+and\s+(.+)$", r"\1 to \2", value)
    bounds = re.split(r"\s+to\s+|\s*[–—]\s*|(?<=\d)\s*-\s*(?=\$?\s*\d)|\s+-\s+", value)
    if len(bounds) == 2:
        lower, upper = map(_amount, bounds)
        if lower is not None and upper is not None and lower <= upper:
            return lower / 2 + upper / 2
    return np.nan


def numeric_answers(answers):
    """Parse the rating and price fields prior to imputation."""
    numeric = pd.DataFrame(index=answers.index)
    ease = pd.to_numeric(answers[EASE_COL], errors="coerce")
    numeric[EASE_COL] = ease.where(ease.between(0, 10))
    for field in LIKERT_COLS:
        numeric[field] = answers[field].map(LIKERT)
    numeric[PRICE] = answers[PRICE].map(parse_ticket_price)
    return numeric.astype(float)


def selections(answers):
    """Return exact normalized options for each multi-select answer."""
    return answers.fillna("").map(
        lambda answer: frozenset(option.strip() for option in answer.split(",") if option.strip())
    )


PARAMETERS_FILE = Path(__file__).resolve().with_name("naive_bayes_parameters.json")
TOKEN_RE = re.compile(TOKEN_PATTERN)
CONTINUOUS = (EASE_COL, PRICE)


def _prepare_features(frame, state):
    """Apply the saved imputation, option lists, and binary word features."""
    answers = normalize_answers(frame, state["column_names"])
    numeric = numeric_answers(answers)
    filled = numeric.fillna(state["imputation"])

    missing = answers.isna()
    missing[list(NUMERIC)] = numeric.isna()
    flags = missing.astype(float).add_suffix("__missing")
    flags["all_answers_missing"] = answers.isna().all(axis=1).astype(float)

    indicators = {}
    for field, options in state["options"].items():
        chosen = selections(answers[field])
        for option in options:
            indicators[f"{field}__option__{option}"] = chosen.map(
                lambda values: float(option in values)
            )

    # Match the trained CountVectorizer's lowercase Unicode word analyzer.
    for field, terms in state["vocabularies"].items():
        words = answers[field].fillna("").map(
            lambda text: set(TOKEN_RE.findall(text.lower()))
        )
        for term in terms:
            indicators[f"{field}__token__{term}"] = words.map(
                lambda values: float(term in values)
            )

    encoded = pd.DataFrame(indicators, index=frame.index, dtype=float)
    return pd.concat([filled, flags, encoded], axis=1)


def _decision_scores(features, model):
    """Score Gaussian, categorical, and binary features with the saved model."""
    continuous = features.loc[:, list(CONTINUOUS)].to_numpy(dtype=float, copy=True)
    continuous[:, 1] = np.log1p(continuous[:, 1])
    categorical = features.loc[:, LIKERT_COLS].to_numpy(dtype=int) - 1
    binary = features.loc[:, model["binary_columns"]].to_numpy(dtype=float)

    means = np.asarray(model["means"])
    variances = np.asarray(model["variances"])
    categorical_log_probs = np.asarray(model["categorical_log_probs"])
    log_present = np.asarray(model["binary_log_present"])
    log_absent = np.asarray(model["binary_log_absent"])
    weights = np.asarray(model["binary_weights"])
    scores = np.empty((len(features), len(model["classes"])))

    for index, log_prior in enumerate(model["log_prior"]):
        variance = variances[index]
        squared_distance = (continuous - means[index]) ** 2
        gaussian = -0.5 * (np.log(2 * np.pi * variance) + squared_distance / variance)
        likert = categorical_log_probs[index][np.arange(len(LIKERT_COLS)), categorical]
        bernoulli = binary * log_present[index] + (1 - binary) * log_absent[index]
        scores[:, index] = (
            log_prior + gaussian.sum(axis=1) + likert.sum(axis=1)
            + (bernoulli * weights).sum(axis=1)
        )
    return scores


def predict_all(filename):
    """Return one sport name per raw CSV row, in the same order as the input."""
    frame = pd.read_csv(filename, keep_default_na=False)
    if frame.empty:
        return []
    with PARAMETERS_FILE.open(encoding="utf-8") as parameter_file:
        bundle = json.load(parameter_file)
    if bundle["format_version"] != 1:
        raise ValueError("Unsupported Naive Bayes parameter file version.")
    features = _prepare_features(frame, bundle["preprocessing"])
    scores = _decision_scores(features, bundle["model"])
    return np.asarray(bundle["model"]["classes"])[scores.argmax(axis=1)].tolist()
