import math
import unittest
from pathlib import Path

import numpy as np

from plasma_level48 import (
    calibrated_capture_probability,
    evaluate_design,
    load_yaml,
    lower_hybrid_omega,
    required_trapping_potential,
    resonant_B,
)


class Level48Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml(Path("config_level48.yaml"))
        cls.vector = np.array([
            4.0, 4.0, 4.0, 3.0, 1.5,
            0.30, 0.30, 0.40, 0.80, 1.10,
        ])

    def test_trapping_potential_has_expected_one_keV_scale(self):
        value = required_trapping_potential(200.0, 1000.0)
        self.assertAlmostEqual(value, 305.57, delta=0.02)

    def test_capture_surrogate_is_monotone(self):
        values = [calibrated_capture_probability(x) for x in np.linspace(0.0, 5.0, 101)]
        self.assertTrue(np.all(np.diff(values) >= 0.0))
        self.assertEqual(values[0], 0.0)

    def test_resonant_field_inverts_dispersion(self):
        omega = 2.0 * math.pi * 2.0e6
        field = resonant_B(omega, 1.0e15)
        self.assertAlmostEqual(lower_hybrid_omega(field, 1.0e15) / omega, 1.0, places=9)

    def test_power_closure_is_exact_by_construction(self):
        result = evaluate_design(self.vector, 200.0, self.cfg)
        direct = float((result.stages.damping_power_W + result.stages.ion_loading_W).sum())
        self.assertAlmostEqual(direct, result.total_power_W, places=10)
        self.assertGreater(result.analytic_tail_fraction, 0.0)
        self.assertLessEqual(result.analytic_tail_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
