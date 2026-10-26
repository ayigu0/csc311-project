from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



# Adapted from the course baseline to use our short column names.
# Declaring Variables for the data file path, column names, and answers.
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
MISSING_VALUES = ["N/A", "N/a", "n/a", "NA", "N/A, since I know nothing about it at all", "Idk",
    "idk",
    "I don't know",
    "i don't know",
    "I don’t know.",
    "I don't know.",
    "not sure",
    "I don't know any songs.",
    "None I do not know of any.",
    "None, I basically know nothing about the sport", ".",
    "-", "None",
    "none",
    "Nothing",
    "nothing",
    "nil",
    "NIL",
    "nothing really",
    "Nothing in particular",
    "nothing particular",
    "nothing in particular",
    "Nothing Specific",
    "None in particular",
    "None really",
    "None in particular.",
    "Nothing in particular.",
    "None, the sport is not popular", "Nothing since I don't really watch it.",
    "Nothing since I don't watch it.",
    "Nothing I don't really watch it.",
    "None I do not watch sports.",
    "None I dont watch sports.", np.nan]

meaningful_responses = [
    "none. maybe hotdogs",
    "Nothing more than 400",
    "Nothing, I'm not really a fan, perhaps a small amount ($100 max) if with friends.",
    "Nothing in particular, something with a good beat?",
    "Nothing, maybe some contempt",
    "i don't know what's going on",
    "I don't know the rules very well but it's fun to watch and learn.",
    "I don't know the rules, so I don't understand it, but I do know that everyone pays close attention when they start running.",
]

ZERO_PRICE_VALUES = [
    "None",
    "none",
    "Nothing",
    "not really willing to",
    "No willing.",
    "Not willing",
    "Not willing.",
    "Unwilling",
    "I would not pay to watch in person",
    "I would not pay for a [this sport] game",
    "I would not pay to watch this sport",
    "Zero. I have no any interest in team sport like [this sport]",
    "Zero. I have no interest in team sport",
    "Zero",
    "None I do not watch sports.",
    "None I dont watch sports.",
    "No I don’t",
    "I wouldnt pay for this sport as I don't enjoy it",
    "You need to pay me to sit there.",
    "The tickets are way too hard to grab, so forget it—none of my friends even play [this sport] anyway."
]

LOCATIONS = [
    "Indoor facility",
    "Outdoor public space",
    "School or university facility",
    "Community recreation facility",
    "Private sports facility",
    "At home or nearby",
    "Others",
]

SEASONS = [
    "Winter",
    "Spring",
    "Summer",
    "Fall",
]

MUSIC_GENRES = [
    "Pop",
    "Rock",
    "Hip-hop or rap",
    "Electronic or dance",
    "Country",
    "Latin",
    "Classical or jazz",
    "Folk or traditional",
    "Other",
    "No particular type",
]

# Returns the dataset as a pandas DataFrame: something like an excel spreadsheet. The first row is treated as the heading.
def load_dataset():
    """Load the project dataset and return a pandas DataFrame."""
    return pd.read_csv(DATA_FILE)

# Renames the columns of the DataFrame to use shorter names.
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
    features = pd.DataFrame(index=data.index) # Create a new DataFrame with the same index as the input DataFrame to hold the features.
    features[EASE_COL] = pd.to_numeric(data[EASE_COL], errors="coerce") #reads that column, converts its values into numbers, and adds it to features, an unconvertible answer becomes a missing value.

    for column in LIKERT_COLS: # Convert agreement answers into numbers
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

# Using the one-hot encoded categories to convert the multi-selection column to a number;
def multi_hot_encode(df, column, categories):
    for category in categories:
        df[category] = np.nan

    for index, value in df[column].items():
        if pd.isna(value):
            continue

        answers = [answer.strip() for answer in value.split(",")]

        for category in categories:
            if category in answers:
                df.loc[index, category] = 1
            else:
                df.loc[index, category] = 0

    return df

# Converting all the missing data in each column with NaN.
def change_missing_to_nan(df):
    """Convert missing values to NaN for easier analysis. Apadted from the template.
    """
    # df.loc[row_label, "column_name"] -- used to change a specific value in a DataFrame.
    count_total = 0
    for_each_column = []
    percent_missing = []
    for column in df.columns:
        index = 0
        count = 0
        for value in df[column]:
            if column == "ticket_price" and value in ZERO_PRICE_VALUES:
                df.loc[index, column] = 0 # None in price column means means 0
            elif value in MISSING_VALUES:
                df.loc[index, column] = np.nan
                count_total += 1
                count += 1
            index += 1
        for_each_column.append(count)
        percent_missing.append((count / (index + 1)) * 100)
    print(f"Changed {count_total} missing values to NaN.")
    print(for_each_column)
    print(percent_missing)
    return df

def calculate_missing_values(df):
    """Calculate the number of missing values in each column of the DataFrame."""
    missing_values = df.isna().sum().sum()
    return missing_values


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

def missing_values_by_label(df, column):
    """Return missing counts and percentages for a column, grouped by label."""
    results = []

    for label, group in df.groupby("label"):
        total_rows = len(group)
        missing_count = group[column].isna().sum()
        missing_percentage = (missing_count / total_rows) * 100

        results.append({
            "label": label,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_percentage": round(missing_percentage, 2)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = load_dataset() # dataset loaded
    df = rename_features(df) # dataset renamed
    df = change_missing_to_nan(df) # dataset missing values changed to NaN
    song_missing = missing_values_by_label(df, "associated_song")
    print(song_missing)
    df = multi_hot_encode(df, "play_locations", LOCATIONS)
    df = multi_hot_encode(df, "play_seasons", SEASONS)
    df = multi_hot_encode(df, "music_genres", MUSIC_GENRES)
