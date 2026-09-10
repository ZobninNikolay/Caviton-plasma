#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

from scipy.constants import e, m_e, pi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasma_extrema import pump_charge_time, wavebreaking_ponderomotive_bound_eV


class ExtremaTests(unittest.TestCase):
    def test_wavebreaking_bound_matches_direct_definition(self):
        temperature = 6.519578
        kappa = 0.210
        omega = 2.0 * pi * 1.224690e9
        thermal_speed = math.sqrt(e * temperature / m_e)
        phase_speed = thermal_speed * math.sqrt(1.0 + 3.0 * kappa**2) / kappa
        field = m_e * omega * phase_speed / e
        direct = e * field**2 / (4.0 * m_e * omega**2)
        self.assertAlmostEqual(
            wavebreaking_ponderomotive_bound_eV(temperature, kappa), direct, places=10
        )

    def test_pump_charge_time_reconstructs_energy(self):
        power = 0.5
        gamma = 2.0e6
        required = 0.25 * power / (2.0 * gamma)
        time = pump_charge_time(required, power, gamma)
        reconstructed = power / (2.0 * gamma) * (1.0 - math.exp(-2.0 * gamma * time))
        self.assertAlmostEqual(reconstructed / required, 1.0, places=12)

    def test_unreachable_energy_returns_infinity(self):
        self.assertTrue(math.isinf(pump_charge_time(1.0, 0.1, 1.0)))


if __name__ == "__main__":
    unittest.main()

