from pathlib import Path
import numpy as np
import pandas as pd


# Adapted from the course baseline to use our short column names.
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "csc311_project_training_202609.csv"
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


def load_dataset():
    """Load the project dataset and return a pandas DataFrame."""
    return pd.read_csv(DATA_FILE)


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


def make_features(df):
    """Build the baseline's numeric features from raw or renamed columns.

    Adapted from project_baseline.py. This is a starting subset of the survey:
    beginner ease, three Likert questions, and four season indicators. Other
    features still need investigation. Missing values are retained for review;
    this function does not drop rows, impute values, or scale features.
    """
    data = rename_features(df)
    features = pd.DataFrame(index=data.index)
    features[EASE_COL] = pd.to_numeric(data[EASE_COL], errors="coerce")

    for column in LIKERT_COLS:
        features[column] = data[column].astype("string").str.strip().map(LIKERT_MAP)

    # Trim spaces around selections so "Spring, Summer" is handled correctly.
    season_answers = data[SEASON_COL].astype("string").str.strip()
    answered = season_answers.notna() & season_answers.ne("")
    season_lists = season_answers.fillna("").str.split(",").map(
        lambda answers: [answer.strip() for answer in answers]
    )
    for season in SEASONS:
        selected = season_lists.map(lambda answers: int(season in answers))
        # A blank response is unknown, rather than four definite zeroes.
        features[f"season_{season.lower()}"] = selected.where(answered)

    return features


def split_by_student(ids, test_frac=0.2, n_folds=5, seed=42, fold_seed=43):
    """Return a test mask and (train, validation) masks for each development fold.

    All masks align with the original rows. Shuffle sorted student IDs so
    membership is reproducible even if the survey rows are reordered.
    """
    if not 0 < test_frac < 1:
        raise ValueError("The test fraction must be between zero and one.")
    if n_folds < 2:
        raise ValueError("At least two development folds are required.")
    if ids.isna().any() or ids.astype("string").str.strip().eq("").any():
        raise ValueError("Every row must have a student ID before splitting.")

    students = np.random.default_rng(seed).permutation(sorted(ids.unique()))
    n_test = int(len(students) * test_frac)
    if n_test == 0 or len(students) - n_test < n_folds:
        raise ValueError("Too few students for a nonempty test set and development folds.")

    test = ids.isin(students[:n_test])
    development_students = np.random.default_rng(fold_seed).permutation(students[n_test:])
    folds = []
    for validation_students in np.array_split(development_students, n_folds):
        validation = ids.isin(validation_students)
        train = ~test & ~validation
        folds.append((train, validation))
    return test, folds


if __name__ == "__main__":
    data = rename_features(load_dataset())
    test, folds = split_by_student(data["id"])
    development = data.loc[~test]
    print(f"Development: {development['id'].nunique()} students, {len(development)} responses")
    print(f"Test: {data.loc[test, 'id'].nunique()} students, {test.sum()} responses")
    for number, (train, validation) in enumerate(folds, start=1):
        print(
            f"Fold {number}: "
            f"train {data.loc[train, 'id'].nunique()} students / {train.sum()} responses, "
            f"validation {data.loc[validation, 'id'].nunique()} students / "
            f"{validation.sum()} responses"
        )
