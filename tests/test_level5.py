#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.constants import atomic_mass, e, pi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasma_level5 import (
    ParticleState,
    SpeciesTable,
    particle_energy_eV,
    push_particles,
    weighted_quantile,
)


class Level5Tests(unittest.TestCase):
    def setUp(self):
        self.species = SpeciesTable(
            names=("Dplus", "D2plus", "D3plus"),
            fractions=np.array([0.1, 0.1, 0.8]),
            masses_kg=np.array([2.014, 4.028, 6.042]) * atomic_mass,
            deuterons_per_ion=np.array([1, 2, 3]),
        )

    def test_molecular_energy_is_reported_per_deuteron(self):
        speed = 1.0e4
        state = ParticleState(
            np.zeros(3), np.zeros(3), np.full(3, speed), np.zeros(3), np.zeros(3),
            np.array([0, 1, 2]),
        )
        _, deuteron = particle_energy_eV(state, self.species)
        self.assertLess(np.ptp(deuteron) / deuteron.mean(), 1.0e-12)

    def test_magnetic_rotation_preserves_energy_without_electric_field(self):
        state = ParticleState(
            np.array([0.01]), np.array([0.0]), np.array([1000.0]), np.array([2000.0]),
            np.array([300.0]), np.array([0]),
        )
        before = particle_energy_eV(state, self.species)[0][0]
        push_particles(state, np.zeros(32), np.linspace(0.0, 0.05, 32), 1.0e-9, 0.01, self.species)
        after = particle_energy_eV(state, self.species)[0][0]
        self.assertLess(abs(after / before - 1.0), 1.0e-12)

    def test_direct_hf_quiver_is_sub_eV(self):
        field = 2.4e5
        omega = 2.0 * pi * 1.22469e9
        quiver_eV = e * field**2 / (4.0 * self.species.masses_kg[0] * omega**2)
        self.assertLess(quiver_eV, 0.02)

    def test_weighted_quantile(self):
        value = weighted_quantile(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 8.0]), [0.5])[0]
        self.assertGreater(value, 2.0)


if __name__ == "__main__":
    unittest.main()
