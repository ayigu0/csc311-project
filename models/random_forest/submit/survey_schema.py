"""Survey field names used by the selected RF preprocessing."""

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
