"""Synthetic unit fixtures only, not benchmark observations."""

import math
import unittest

from evidence_state_repro import mmqa_dev_historical as dev
from evidence_state_repro import mmqa_endpoint_historical as endpoint
from evidence_state_repro import m3_frozen as m3
from evidence_state_repro.selection import fixed_budget


class HistoricalScoringTests(unittest.TestCase):
    def test_casefold_and_article(self):
        self.assertEqual(endpoint.best_score("The CAT!", ["cat"]), (1, 1.0))

    def test_member_match_is_not_list_coverage(self):
        self.assertEqual(endpoint.best_score("cat", ["cat", "dog"]), (1, 1.0))

    def test_hyphen_versions_remain_distinct(self):
        self.assertEqual(dev.norm("north-east"), "north east")
        self.assertEqual(endpoint.normalize_answer("north-east"), "northeast")
        self.assertNotEqual(dev.token_f1("north-east", "north east"), endpoint.token_f1("north-east", "north east"))

    def test_empty_token_f1(self):
        self.assertEqual(endpoint.token_f1("", ""), 1.0)

    def test_m3_partial_credit_and_strict_full_match(self):
        score = m3.official_accuracy(["cat"], [["cat"], ["dog"]])
        self.assertEqual(score, 0.5)
        self.assertNotEqual(score, 1.0)

    def test_m3_covered_group_count_contract(self):
        self.assertEqual(m3.official_accuracy(["cat", "cat"], [["cat"]]), 0.5)

    def test_m3_empty_prediction(self):
        self.assertEqual(m3.official_accuracy([], [["cat"]]), 0.0)

    def test_m3_parser_does_not_execute_code(self):
        parsed = m3.convert_to_list("['cat', 'dog']")
        self.assertEqual(parsed, ["cat", "dog"])


class FixedBudgetTests(unittest.TestCase):
    def test_ceiling_budget_and_qid_tie(self):
        self.assertEqual(fixed_budget(["b", "a", "c"], {"a": 1, "b": 1, "c": 0}), ["a"])

    def test_forced_cases_occupy_budget(self):
        qids = [str(i) for i in range(10)]
        self.assertEqual(fixed_budget(qids, {q: int(q) for q in qids}, ["0"]), ["0", "9"])

    def test_too_many_forced_fails(self):
        with self.assertRaises(ValueError):
            fixed_budget(["a", "b", "c"], {}, ["a", "b"])

    def test_nan_score_fails(self):
        with self.assertRaises(ValueError):
            fixed_budget(["a"], {"a": math.nan})

    def test_duplicate_qids_fail(self):
        with self.assertRaises(ValueError):
            fixed_budget(["a", "a"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
