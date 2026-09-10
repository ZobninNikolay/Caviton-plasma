import math
import unittest

from scipy.constants import e

from plasma_level46 import (
    M_D,
    double_layer_solution,
    lower_hybrid_omega,
    magnetic_field_for_lower_hybrid,
    reflected_deuteron_energy_eV,
)


class Level46Tests(unittest.TestCase):
    def test_hot_electrons_raise_double_layer_potential(self):
        cold = double_layer_solution(0.001, 300.0, 8.0, M_D)
        hot = double_layer_solution(0.03, 300.0, 8.0, M_D)
        self.assertTrue(cold["valid"] and hot["valid"])
        self.assertGreater(hot["potential_V"], cold["potential_V"])

    def test_lower_hybrid_field_inversion(self):
        ne = 4.0e16
        omega = 2.0 * math.pi * 1.0e6
        field = magnetic_field_for_lower_hybrid(omega, ne, (5.0e-4, 0.2))
        self.assertAlmostEqual(lower_hybrid_omega(field, ne) / omega, 1.0, places=10)

    def test_reflection_energy_is_one_keV_near_155_km_s(self):
        energy = reflected_deuteron_energy_eV(155.0e3)
        self.assertAlmostEqual(energy / 1000.0, 1.0, delta=0.02)

    def test_deuteron_one_keV_velocity(self):
        velocity = math.sqrt(2.0 * e * 1000.0 / M_D)
        self.assertAlmostEqual(velocity / 1.0e3, 309.5, delta=0.5)


if __name__ == "__main__":
    unittest.main()
