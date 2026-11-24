"""Regression checks for parsing, row alignment, and training-only fitting."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.random_forest.data_preprocessing import (
    ANSWER_COLS, fit_preprocessor, parse_ticket_price, transform_features,
)


class PreprocessingTests(unittest.TestCase):
    def test_ticket_amounts_ranges_and_zero(self):
        cases = {
            "$1,000": 1000, "$10 000": 10000, "1k": 1000,
            "200CAD": 200, "200dollars": 200, "around $50.": 50,
            "CAD $80–$120": 100, "$100 - $300": 200,
            "$2000 to $500": 1250, "50 to 150 CAD": 100,
            "Forty five dollars": 45, "none": 0, "Not willing.": 0,
            "I would not pay for this sport": 0,
            "Nothing, maybe $100 with friends": 100,
        }
        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(parse_ticket_price(answer)[0], expected)
        self.assertEqual(parse_ticket_price("CAD $80–$120")[1], 1)

    def test_ambiguous_or_invalid_prices_are_not_guessed(self):
        for answer in [None, "N/A", "$-50", "-20", "1€", "USD 100",
                       "$200, $500 for a great game", "1 to 10 scale probably 1",
                       "100 to 200 CAD, at most 1000 CAD", "with 3 friends"]:
            with self.subTest(answer=answer):
                self.assertTrue(np.isnan(parse_ticket_price(answer)[0]))

    def training_data(self):
        data = pd.DataFrame("", index=[10, 30, 90], columns=ANSWER_COLS)
        data["beginner_ease"] = [2, 10, ""]
        data["strategy_importance"] = ["Agree", "agree", "N/A"]
        data["ticket_price"] = ["$50", "$150", ""]
        data["associated_song"] = ["Shared tune", "shared tune", "singleton"]
        data["play_seasons"] = ["Spring, Summer", "Summer", ""]
        return data

    def test_frozen_training_rules_work_on_label_free_unseen_rows(self):
        train = self.training_data()
        state = fit_preprocessor(train)
        frozen = json.dumps(state, sort_keys=True)
        unseen = pd.DataFrame("", index=[7, 101], columns=ANSWER_COLS)
        unseen["associated_song"] = ["UNSEENONLY", "shared tune"]
        unseen["ticket_price"] = ["$999999", ""]
        unseen["beginner_ease"] = [99, ""]
        unseen["play_seasons"] = ["Summer, Monsoon", ""]
        features = transform_features(unseen, state)
        self.assertEqual(features.index.tolist(), [7, 101])
        self.assertEqual(features.columns.tolist(), state["feature_columns"])
        self.assertEqual(features.loc[7, "beginner_ease"], 6)
        self.assertEqual(features.loc[101, "ticket_price_cad"], 100)
        self.assertEqual(features.loc[7, "beginner_ease__missing"], 1)
        self.assertEqual(features.loc[7, "play_seasons__summer"], 1)
        self.assertEqual(features.loc[7, "play_seasons__unknown_option"], 1)
        self.assertEqual(features.loc[101, "play_seasons__missing"], 1)
        self.assertEqual(features.loc[101, "associated_song__word_shared"], 1)
        self.assertNotIn("unseenonly", state["vocabularies"]["associated_song"])
        self.assertNotIn("singleton", state["vocabularies"]["associated_song"])
        self.assertEqual(json.dumps(state, sort_keys=True), frozen)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_id_and_labels_do_not_affect_features(self):
        train = self.training_data()
        decorated = train.assign(id=[1, 2, 3], label=["Soccer", "Baseball", "Basketball"])
        state = fit_preprocessor(train)
        self.assertEqual(state, fit_preprocessor(decorated))
        pd.testing.assert_frame_equal(transform_features(train, state), transform_features(decorated, state))

    def test_empty_rows_are_retained_and_flagged(self):
        train = self.training_data()
        empty = pd.DataFrame("", index=[500], columns=ANSWER_COLS)
        output = transform_features(empty, fit_preprocessor(train))
        self.assertEqual(output.index.tolist(), [500])
        self.assertEqual(output.loc[500, "all_answers_missing"], 1)
        self.assertTrue(np.isfinite(output.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
