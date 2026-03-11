import unittest

import numpy as np

from impact_model.execution import (
    feasible_profile_schedule,
    make_volume_profile,
    optimize_schedule_single_asset,
    schedule_cost_components,
)


class Test_ExecutionOptimization(unittest.TestCase):
    def setUp(self):
        self.params = np.array([0.25, 0.28, 0.42, 0.08, 0.09, 0.35, 0.04, 0.7, 0.02, -0.01, 0.01, 0.0, 0.01, -0.01], dtype=float)
        self.Q = 125000.0
        self.V_day = 2.0e7
        self.sigma_day = 0.018
        self.liq_z_day = -0.2
        self.vol_z_day = 0.15
        self.max_participation = 0.25
        self.n_slices = 13
        self.lambda_risk = 1e-6

    def test_schedule_is_feasible_and_conserves_quantity(self):
        schedule, costs = optimize_schedule_single_asset(
            self.Q,
            self.V_day,
            self.sigma_day,
            self.params,
            self.liq_z_day,
            self.vol_z_day,
            lambda_risk=self.lambda_risk,
            max_participation=self.max_participation,
            n_slices=self.n_slices,
        )
        profile = make_volume_profile(self.n_slices)
        upper = self.max_participation * self.V_day * profile
        q_abs_target = min(abs(self.Q), 0.98 * float(np.sum(upper)))

        self.assertEqual(schedule.shape, (self.n_slices,))
        self.assertTrue(np.all(np.isfinite(schedule)))
        self.assertTrue(np.isfinite(costs["total"]))
        self.assertAlmostEqual(float(np.sum(np.abs(schedule))), q_abs_target, delta=1e-4)
        self.assertTrue(np.all(np.abs(schedule) <= upper + 1e-6))
        self.assertTrue(np.all(np.sign(schedule[np.abs(schedule) > 0]) == np.sign(self.Q)))

    def test_optimizer_improves_or_matches_feasible_seed_cost(self):
        profile = make_volume_profile(self.n_slices)
        upper = self.max_participation * self.V_day * profile
        q_abs_target = min(abs(self.Q), 0.98 * float(np.sum(upper)))
        seed = feasible_profile_schedule(q_abs_target, upper, profile)
        seed_cost = schedule_cost_components(
            seed,
            np.sign(self.Q) * q_abs_target,
            self.V_day,
            self.sigma_day,
            self.params,
            self.liq_z_day,
            self.vol_z_day,
            self.lambda_risk,
        )["total"]

        schedule, costs = optimize_schedule_single_asset(
            self.Q,
            self.V_day,
            self.sigma_day,
            self.params,
            self.liq_z_day,
            self.vol_z_day,
            lambda_risk=self.lambda_risk,
            max_participation=self.max_participation,
            n_slices=self.n_slices,
        )
        self.assertTrue(np.isfinite(costs["total"]))
        self.assertLessEqual(costs["total"], seed_cost + 1e-6)


if __name__ == "__main__":
    unittest.main()
