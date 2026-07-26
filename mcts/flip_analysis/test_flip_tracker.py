import unittest

import numpy as np
from flip_analysis.search_with_snapshots import (
    _flip_tracking_start_sim,
    _significant_top_candidate,
)


class SignificantTopCandidateTests(unittest.TestCase):
    def test_tracking_starts_after_one_visit_is_smaller_than_margin(self) -> None:
        self.assertEqual(_flip_tracking_start_sim(.05), 21)
        self.assertEqual(_flip_tracking_start_sim(.10), 11)
        self.assertLess(1 / _flip_tracking_start_sim(.05), .05)
        self.assertGreaterEqual(1 / (_flip_tracking_start_sim(.05) - 1), .05)

    def test_tracking_margin_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            _flip_tracking_start_sim(0)

    def test_first_flip_matches_raw_to_search_relative_shift(self) -> None:
        # Raw: A=.20, B=.18, C=.14. Search: A=.21, B=.24, C=.23.
        # B moved from .02 behind A to .03 ahead: exactly .05 total.
        baseline = np.array([.20, .18, .14, .12, .12, .12, .12])
        ns = np.array([21, 24, 23, 8, 8, 8, 8])

        result = _significant_top_candidate(
            ns, total=100, baseline=baseline, winner_pos=0, margin=.05
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 1)
        self.assertAlmostEqual(result[1], .05)

    def test_snapshot_prevents_immediate_flip_back(self) -> None:
        # At the B snapshot A is .03 behind. Merely becoming the instantaneous
        # top move is insufficient; A must reach .02 ahead for a fresh .05 shift.
        baseline = np.array([.21, .24, .23, .08, .08, .08, .08])
        barely_top = np.array([241, 240, 230, 73, 72, 72, 72])
        qualified = np.array([260, 240, 220, 70, 70, 70, 70])

        self.assertIsNone(_significant_top_candidate(
            barely_top, 1000, baseline, winner_pos=1, margin=.05
        ))
        result = _significant_top_candidate(
            qualified, 1000, baseline, winner_pos=1, margin=.05
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], .05)

    def test_close_third_candidate_needs_a_fresh_margin(self) -> None:
        # C is only .01 behind B at the snapshot, so it must become .04 ahead.
        baseline = np.array([.21, .24, .23, .08, .08, .08, .08])
        tied = np.array([220, 240, 240, 75, 75, 75, 75])
        qualified = np.array([210, 240, 280, 68, 68, 67, 67])

        self.assertIsNone(_significant_top_candidate(
            tied, 1000, baseline, winner_pos=1, margin=.05
        ))
        result = _significant_top_candidate(
            qualified, 1000, baseline, winner_pos=1, margin=.05
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 2)
        self.assertAlmostEqual(result[1], .05)

    def test_catching_up_from_far_behind_can_flip_at_a_tie(self) -> None:
        # C moved from .06 behind B to tied for the current top: the .06
        # relative displacement is significant even without a strict lead.
        baseline = np.array([.20, .24, .18, .095, .095, .095, .095])
        tied = np.array([210, 230, 230, 83, 83, 82, 82])

        result = _significant_top_candidate(
            tied, 1000, baseline, winner_pos=1, margin=.05
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 2)
        self.assertAlmostEqual(result[1], .06)

    def test_candidate_can_move_from_baseline_tie_to_significant_lead(self) -> None:
        baseline = np.array([.25, .25, .10, .10, .10, .10, .10])
        ns = np.array([200, 250, 110, 110, 110, 110, 110])

        result = _significant_top_candidate(
            ns, 1000, baseline, winner_pos=0, margin=.05
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 1)
        self.assertAlmostEqual(result[1], .05)

    def test_tied_top_candidate_with_largest_displacement_wins(self) -> None:
        baseline = np.array([.30, .25, .15, .10, .10, .10])
        tied = np.array([200, 250, 250, 100, 100, 100])

        result = _significant_top_candidate(
            tied, 1000, baseline, winner_pos=0, margin=.05
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 2)
        self.assertAlmostEqual(result[1], .20)


if __name__ == "__main__":
    unittest.main()
