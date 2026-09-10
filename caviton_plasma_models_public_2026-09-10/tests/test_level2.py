#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasma_level2 import (
    build_species,
    find_least_damped_slow_candidate,
    langmuir_long_wavelength_asymptotic,
    load_yaml,
    longitudinal_dielectric,
    solve_kinetic_mode,
)


class Level2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml(ROOT / "config_level2.yaml")
        cls.background = {
            "pressure_Pa": 0.05,
            "absorbed_power_W": 15.0,
            "electron_density_m3": 1.6212010744103302e16,
            "electron_temperature_eV": 6.519577709856203,
            "nu_in_scaled_s-1": 1.992799446707278e5,
            "nu_en_scaled_s-1": 1.7043620219985612e6,
        }

    def test_fractions_and_quasineutrality(self):
        composition = self.cfg["composition_scenarios"][1]
        electron, ions = build_species(self.background, 0.1, composition)
        self.assertAlmostEqual(sum(item.density_m3 for item in ions), electron.density_m3, delta=1.0)

    def test_langmuir_root_residual(self):
        composition = self.cfg["composition_scenarios"][1]
        electron, ions = build_species(self.background, 0.1, composition)
        k = 0.21 / electron.debye_length_m
        omega, residual = solve_kinetic_mode("langmuir", k, electron, ions, self.cfg["solver"])
        self.assertLess(residual, 1.0e-7)
        self.assertGreater(omega.real, 0.0)
        self.assertLess(omega.imag, 0.0)

    def test_long_wavelength_langmuir_asymptotic_is_finite(self):
        composition = self.cfg["composition_scenarios"][1]
        electron, _ = build_species(self.background, 0.1, composition)
        k_m1 = 0.01 / electron.debye_length_m
        omega, indicator = langmuir_long_wavelength_asymptotic(k_m1, electron)
        self.assertTrue(np.isfinite(omega.real))
        self.assertGreater(omega.real, electron.plasma_frequency_rad_s)
        self.assertLessEqual(omega.imag, 0.0)
        self.assertLess(indicator, 1.0e-10)

    def test_acoustic_root_is_below_langmuir_root(self):
        composition = self.cfg["composition_scenarios"][1]
        electron, ions = build_species(self.background, 0.1, composition)
        k = 0.21 / electron.debye_length_m
        acoustic, residual = solve_kinetic_mode("ion_acoustic_fast", k, electron, ions, self.cfg["solver"])
        langmuir, _ = solve_kinetic_mode("langmuir", k, electron, ions, self.cfg["solver"])
        self.assertLess(residual, 1.0e-7)
        self.assertLess(acoustic.real, 0.01 * langmuir.real)
        self.assertGreater(acoustic.real, 0.0)

    def test_slow_candidate_is_not_promoted_to_propagating_mode(self):
        composition = self.cfg["composition_scenarios"][1]
        result = find_least_damped_slow_candidate(self.background, 0.1, 0.21, composition, self.cfg["solver"])
        self.assertIsNotNone(result)
        self.assertFalse(result["propagating_Q_ge_threshold"])


if __name__ == "__main__":
    unittest.main()
