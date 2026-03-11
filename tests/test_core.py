from __future__ import annotations

import unittest

import numpy as np

from src.clt_playground.core import build_progression_sizes, simulate_clt


class CoreTests(unittest.TestCase):
    def test_theoretical_standard_error_matches_known_bernoulli_case(self) -> None:
        simulation = simulate_clt(
            distribution_name="bernoulli",
            params={"p": 0.25},
            sample_size=100,
            num_simulations=500,
            seed=7,
        )
        expected = np.sqrt(0.25 * 0.75 / 100.0)
        self.assertAlmostEqual(simulation.theoretical_standard_error, expected, places=8)

    def test_progression_sizes_are_sorted_and_unique(self) -> None:
        self.assertEqual(build_progression_sizes(30), [1, 6, 15, 30])
        self.assertEqual(build_progression_sizes(2), [1, 2])


if __name__ == "__main__":
    unittest.main()
