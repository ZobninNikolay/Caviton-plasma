#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasma_level1 import CrossSection, debye_length, load_cross_sections, neutral_density


class SanityTests(unittest.TestCase):
    def test_neutral_density(self):
        self.assertAlmostEqual(neutral_density(0.1, 300.0), 2.414e19, delta=2e16)

    def test_debye_length(self):
        self.assertAlmostEqual(debye_length(1e16, 5.0), 1.662e-4, delta=2e-7)

    def test_cross_sections_and_rates_are_positive(self):
        xs = load_cross_sections(ROOT)
        self.assertGreater(xs["e_momentum"].maxwell_rate(5.0), 0.0)
        self.assertGreater(xs["ionization"].maxwell_rate(5.0), 0.0)
        self.assertTrue(math.isfinite(xs["ion_momentum"](1.0)))


if __name__ == "__main__":
    unittest.main()

