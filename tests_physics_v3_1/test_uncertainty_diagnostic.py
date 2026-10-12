"""Independent linear kinematic uncertainty check; no neural or engine execution."""
import unittest

import numpy as np


class UncertaintyPropagation(unittest.TestCase):
    def test_velocity_innovation_position_covariance(self):
        dt = 1 / 30
        transition = np.array([[1, dt], [0, 1]])
        injection = np.array([[dt], [1]])
        covariance = np.zeros((2, 2))
        for _ in range(2):
            covariance = transition @ covariance @ transition.T + injection @ injection.T
        # Two independent unit-variance velocity innovations: p2=dt*(2e1+e2).
        np.testing.assert_allclose(covariance, [[5 * dt**2, 3 * dt], [3 * dt, 2]])
        self.assertGreater(covariance[0, 0], 2 * dt**2)

    def test_closed_form_matches_independent_monte_carlo(self):
        rng = np.random.default_rng(31013)
        innovations = rng.normal(size=(100000, 8))
        velocity = innovations.cumsum(1)
        position = velocity.cumsum(1) / 30
        expected_position_variance = np.sum(np.arange(1, 9)**2) / 900
        self.assertAlmostEqual(position[:, -1].var() / expected_position_variance, 1, delta=0.02)
        self.assertAlmostEqual(velocity[:, -1].var() / 8, 1, delta=0.02)
