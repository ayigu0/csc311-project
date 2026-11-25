"""Frozen RF survey schema and historical split; independent of team main changes."""

from pathlib import Path
import numpy as np

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "csc311_project_training_202609.csv"
EASE_COL = "beginner_ease"
LIKERT_COLS = ["strategy_importance", "injury_risk", "individual_brilliance"]
SEASON_COL = "play_seasons"
SEASONS = ["Winter", "Spring", "Summer", "Fall"]
LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}


def rename_features(df):
    """Return a DataFrame with short feature names; keep id and label unchanged."""
    column_names = {
        (
            "How easy do you think it would be for someone with no prior experience "
            "to start playing this sport recreationally? Rate it on a scale from 0 "
            "to 10, where 0 means extremely difficult and 10 means extremely easy."
        ): "beginner_ease",
        "Success in this sport depends heavily on strategy and planning.": "strategy_importance",
        "Players must be comfortable with a high risk of injury.": "injury_risk",
        "Individual brilliance can outweigh teamwork in this sport.": "individual_brilliance",
        "Where might you play this sport?": "play_locations",
        (
            "During which seasons would you be most likely to play this sport "
            "recreationally in Toronto?"
        ): "play_seasons",
        "What types of music do you most associate with this sport?": "music_genres",
        (
            "What foods and drinks do you associate with watching this sport in "
            "the place or culture you know best?"
        ): "food_and_drinks",
        "What colour do you associate with this sport?": "associated_colour",
        "Which song do you associate with this sport?": "associated_song",
        "How do you feel when you watch this sport?": "watching_feelings",
        (
            "How much would you be willing to pay for a decent seat at a "
            "professional competition of this sport?"
        ): "ticket_price",
    }
    return df.rename(columns=column_names)

def split_by_student(ids, val_frac=0.2, test_frac=0.3, seed=42):
    """Return train/validation/test masks, keeping each student's rows together.

    Adapted from project_baseline.py. Call this on the full ID column before
    exploration or dropping any rows. Defaults are 50%/20%/30% of students,
    subject to rounding. Sorting IDs makes a fixed seed independent of row
    order. Teammates must use the same dataset, seed, and fractions.
    """
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("Split fractions must be positive and leave a training set.")
    if ids.isna().any() or ids.astype("string").str.strip().eq("").any():
        raise ValueError("Every row must have a student ID before splitting.")

    rng = np.random.default_rng(seed)
    students = rng.permutation(sorted(ids.unique()))
    n_test = int(len(students) * test_frac)
    n_val = int(len(students) * val_frac)
    if n_test == 0 or n_val == 0 or n_test + n_val >= len(students):
        raise ValueError("There are too few students for three nonempty splits.")

    test_ids = students[:n_test]
    val_ids = students[n_test:n_test + n_val]
    train_ids = students[n_test + n_val:]
    return ids.isin(train_ids), ids.isin(val_ids), ids.isin(test_ids)
