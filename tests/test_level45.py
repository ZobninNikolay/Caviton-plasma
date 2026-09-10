import math
import unittest

import numpy as np

from plasma_level45 import cic_deposit, poisson_field


class Level45Tests(unittest.TestCase):
    def test_uniform_deposition_is_uniform(self):
        cells = 64
        particles = 8 * cells
        x = (np.arange(particles) + 0.5) * cells / particles
        weights = np.full(particles, cells / particles)
        density = cic_deposit(x, weights, cells, float(cells))
        self.assertLess(float(np.ptp(density)), 1.0e-12)

    def test_poisson_recovers_sinusoidal_field(self):
        cells = 128
        length = 2.0 * math.pi
        x = (np.arange(cells) + 0.5) * length / cells
        charge = 0.1 * np.cos(x)
        field, _ = poisson_field(1.0 + charge, np.ones(cells), length)
        expected = 0.1 * np.sin(x)
        self.assertLess(float(np.max(np.abs(field - expected))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
