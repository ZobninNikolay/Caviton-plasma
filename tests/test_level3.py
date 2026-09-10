#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasma_level3 import (
    ZakharovCoefficients,
    neutral_stability_field,
    quartic_growth_rate,
)


class Level3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coeff = ZakharovCoefficients(
            electron_density_m3=1.6212e16,
            electron_temperature_eV=6.52,
            lambda_De_m=1.4908e-4,
            carrier_k_m1=1408.7,
            carrier_omega_rad_s=7.695e9,
            carrier_group_velocity_ms=8.032e5,
            langmuir_damping_s1=1.935e6,
            acoustic_kinetic_damping_at_carrier_s1=9.80e4,
            ion_neutral_damping_frequency_s1=1.378e5,
            electron_neutral_frequency_s1=1.704e6,
            effective_ion_mass_kg=4.8336 * 1.66053906660e-27,
            acoustic_speed_ms=1.1667e4,
            dispersion_P_m2s=437.3,
            density_coupling_A_m3s=2.373e-7,
            ponderomotive_B=2.758e14,
        )

    def test_neutral_stability_changes_growth_sign(self):
        K = 2.0 * math.pi / 0.05
        threshold = neutral_stability_field(K, self.coeff)
        self.assertLess(quartic_growth_rate(0.99 * threshold, K, self.coeff), 0.0)
        self.assertGreater(quartic_growth_rate(1.01 * threshold, K, self.coeff), 0.0)

    def test_threshold_is_finite_and_positive(self):
        K = 2.0 * math.pi / 0.05
        threshold = neutral_stability_field(K, self.coeff)
        self.assertTrue(np.isfinite(threshold))
        self.assertGreater(threshold, 0.0)

    def test_field_units_scale_with_damping(self):
        K = 2.0 * math.pi / 0.05
        base = neutral_stability_field(K, self.coeff)
        higher = ZakharovCoefficients(**{**self.coeff.__dict__, "langmuir_damping_s1": 2.0 * self.coeff.langmuir_damping_s1})
        self.assertGreater(neutral_stability_field(K, higher), base)

    def test_analytic_continuum_optimum(self):
        optimum_K = math.sqrt(self.coeff.langmuir_damping_s1 / self.coeff.dispersion_P_m2s)
        optimum = neutral_stability_field(optimum_K, self.coeff)
        self.assertGreater(neutral_stability_field(0.9 * optimum_K, self.coeff), optimum)
        self.assertGreater(neutral_stability_field(1.1 * optimum_K, self.coeff), optimum)


if __name__ == "__main__":
    unittest.main()
