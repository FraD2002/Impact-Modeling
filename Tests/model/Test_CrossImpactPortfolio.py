import unittest

import numpy as np

from impact_model.execution import make_volume_profile, optimize_portfolio_schedule


class Test_CrossImpactPortfolio(unittest.TestCase):
    def test_portfolio_schedule_constraints(self):
        Q_vec = np.array([150000.0, -90000.0, 60000.0], dtype=float)
        V_vec = np.array([1.8e7, 2.2e7, 1.5e7], dtype=float)
        sigma_vec = np.array([0.02, 0.015, 0.022], dtype=float)
        params_vec = np.tile(np.array([0.25, 0.27, 0.4, 0.08, 0.09, 0.33, 0.04, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float), (3, 1))
        liq_z_vec = np.array([-0.2, 0.1, -0.05], dtype=float)
        vol_z_vec = np.array([0.2, -0.1, 0.15], dtype=float)
        cov_matrix = np.array(
            [
                [1.0, 0.25, 0.1],
                [0.25, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ],
            dtype=float,
        )
        cross_matrix = np.array(
            [
                [0.0, 0.02, 0.01],
                [0.02, 0.0, 0.015],
                [0.01, 0.015, 0.0],
            ],
            dtype=float,
        )
        n_slices = 13
        max_participation = 0.25

        schedule = optimize_portfolio_schedule(
            Q_vec,
            V_vec,
            sigma_vec,
            params_vec,
            liq_z_vec,
            vol_z_vec,
            cov_matrix,
            cross_matrix,
            lambda_risk=1e-6,
            lambda_port=1e-6,
            max_participation=max_participation,
            n_slices=n_slices,
        )

        self.assertEqual(schedule.shape, (Q_vec.size, n_slices))
        self.assertTrue(np.all(np.isfinite(schedule)))

        profile = make_volume_profile(n_slices)
        upper = max_participation * np.outer(V_vec, profile)
        for i in range(Q_vec.size):
            q_abs_target = min(abs(Q_vec[i]), 0.98 * float(np.sum(upper[i])))
            self.assertAlmostEqual(float(np.sum(np.abs(schedule[i]))), q_abs_target, delta=1e-4)
            self.assertTrue(np.all(np.abs(schedule[i]) <= upper[i] + 1e-6))
            nonzero = np.abs(schedule[i]) > 0
            self.assertTrue(np.all(np.sign(schedule[i][nonzero]) == np.sign(Q_vec[i])))


if __name__ == "__main__":
    unittest.main()
