"""Shared survey cleaning and train-fitted features for all three models."""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler

from .dataexploration import EASE_COL, LIKERT_COLS, LIKERT_MAP, rename_features

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


def normalize_answers(frame):
    """Standardize raw or renamed survey columns, preserving substantive answers."""
    answers = rename_features(frame).reindex(columns=FIELDS).astype("string")
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


class SurveyPreprocessor:
    """Train on each fold separately. Return aligned named features for each model."""

    def __init__(self, min_text_frequency=2, max_terms_per_field=200):
        self.min_text_frequency = min_text_frequency
        self.max_terms_per_field = max_terms_per_field

    def fit(self, frame):
        if frame.empty:
            raise ValueError("At least one training response is required.")
        answers = normalize_answers(frame)
        numeric = numeric_answers(answers)
        self.imputation_ = numeric.median().fillna(0.0)
        for field in LIKERT_COLS:
            mode = numeric[field].mode()
            self.imputation_[field] = mode.iloc[0] if len(mode) else 3.0
        self.scaler_ = StandardScaler().fit(numeric.fillna(self.imputation_))
        self.options_ = {
            field: sorted(set().union(*selections(answers[field])))
            for field in MULTI
        }
        self.text_encoders_ = {}
        analyzer = CountVectorizer(token_pattern=TOKEN_PATTERN).build_analyzer()
        for field in TEXT:
            texts = answers[field].fillna("")
            frequencies = Counter(term for answer in texts for term in set(analyzer(answer)))
            terms = sorted(
                (term for term, count in frequencies.items() if count >= self.min_text_frequency),
                key=lambda term: (-frequencies[term], term),
            )[:self.max_terms_per_field]
            if terms:
                vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, vocabulary=terms)
                tfidf = TfidfTransformer().fit(vectorizer.transform(texts))
                self.text_encoders_[field] = vectorizer, tfidf
        return self

    def transform(self, frame, model="random_forest"):
        """Scale numeric and use TF-IDF for softmax. Use binary text otherwise."""
        if not hasattr(self, "text_encoders_"):
            raise ValueError("Fit the survey preprocessor before transforming responses.")
        if model not in {"random_forest", "softmax", "naive_bayes"}:
            raise ValueError(f"Unknown model: {model}")
        answers = normalize_answers(frame)
        numeric = numeric_answers(answers)
        filled = numeric.fillna(self.imputation_)
        if model == "softmax" and len(frame):
            filled = pd.DataFrame(self.scaler_.transform(filled), index=frame.index, columns=NUMERIC)

        missing = answers.isna()
        missing[list(NUMERIC)] = numeric.isna()
        flags = missing.astype(float).add_suffix("__missing")
        flags["all_answers_missing"] = answers.isna().all(axis=1).astype(float)
        indicators = {}
        for field, options in self.options_.items():
            chosen = selections(answers[field])
            for option in options:
                indicators[f"{field}__option__{option}"] = chosen.map(lambda values: float(option in values))
        parts = [filled, flags, pd.DataFrame(indicators, index=frame.index, dtype=float)]
        for field, (vectorizer, tfidf) in self.text_encoders_.items():
            columns = [f"{field}__token__{term}" for term in vectorizer.get_feature_names_out()]
            if frame.empty:
                parts.append(pd.DataFrame(index=frame.index, columns=columns, dtype=float))
                continue
            counts = vectorizer.transform(answers[field].fillna(""))
            values = tfidf.transform(counts) if model == "softmax" else (counts > 0)
            parts.append(pd.DataFrame(values.toarray(), index=frame.index, columns=columns, dtype=float))
        return pd.concat(parts, axis=1)
