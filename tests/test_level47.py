import math
import unittest
from pathlib import Path

from scipy.constants import e

from plasma_level47 import (
    M_D,
    driven_field,
    harmonic_for_target,
    load_yaml,
    lower_hybrid_omega,
    resonant_B,
)


class Level47Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_yaml(Path("config_level47.yaml"))

    def test_resonant_field_inverts_lower_hybrid_frequency(self):
        ne = 8.0e14
        omega = 2.0 * math.pi * 1.0e6
        field = resonant_B(omega, ne)
        self.assertAlmostEqual(lower_hybrid_omega(field, ne) / omega, 1.0, places=9)

    def test_first_harmonic_targets_about_one_keV_at_one_MHz(self):
        length = self.cfg["experiment"]["chamber_length_m"]
        harmonic = harmonic_for_target(1.0e6, length, 1000.0)
        velocity = 1.0e6 * length / harmonic
        energy = 0.5 * M_D * velocity**2 / e
        self.assertEqual(harmonic, 1)
        self.assertAlmostEqual(energy / 1000.0, 1.0, delta=0.03)

    def test_driven_field_scales_as_sqrt_power(self):
        first = driven_field(1.0, self.cfg)
        fourth = driven_field(4.0, self.cfg)
        self.assertAlmostEqual(fourth / first, 2.0, places=12)


if __name__ == "__main__":
    unittest.main()
