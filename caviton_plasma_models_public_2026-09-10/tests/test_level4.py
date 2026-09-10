#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plasma_level4 import (
    apply_tridiagonal,
    crank_nicolson_dispersion,
    radial_grid,
    radial_laplacian_coefficients,
    sponge_profile,
)


class Level4Tests(unittest.TestCase):
    def test_radial_laplacian_annihilates_constant(self):
        r, dr = radial_grid(0.05, 96)
        lap = radial_laplacian_coefficients(r, dr)
        residual = apply_tridiagonal(np.ones_like(r), *lap)
        self.assertLess(np.max(np.abs(residual)), 1.0e-8)

    def test_dispersion_step_preserves_constant(self):
        r, dr = radial_grid(0.05, 64)
        lap = radial_laplacian_coefficients(r, dr)
        field = np.full(len(r), 4200.0 + 0.0j)
        result = crank_nicolson_dispersion(field, 1.0e-9, 437.0, lap)
        self.assertLess(np.max(np.abs(result - field)), 1.0e-8)

    def test_dispersion_step_preserves_radial_norm(self):
        r, dr = radial_grid(0.05, 96)
        lap = radial_laplacian_coefficients(r, dr)
        rng = np.random.default_rng(7)
        field = rng.normal(size=len(r)) + 1j * rng.normal(size=len(r))
        result = crank_nicolson_dispersion(field, 2.0e-9, 437.0, lap)
        before = np.sum(r * np.abs(field) ** 2)
        after = np.sum(r * np.abs(result) ** 2)
        self.assertLess(abs(after / before - 1.0), 1.0e-12)

    def test_sponge_is_zero_inside_and_monotonic(self):
        r, _ = radial_grid(0.05, 100)
        sponge = sponge_profile(r, 0.05, 0.8, 3.0e6)
        self.assertEqual(float(sponge[r < 0.04].max()), 0.0)
        self.assertTrue(np.all(np.diff(sponge) >= 0.0))


if __name__ == "__main__":
    unittest.main()
