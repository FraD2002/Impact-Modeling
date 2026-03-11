import unittest

import numpy as np

from impact_model.execution import (
    build_transient_kernel,
    effective_participation_cap,
    front_loaded_schedule,
    pov_schedule,
)


class Test_AdvancedExecutionPolicies(unittest.TestCase):
    def test_hybrid_transient_kernel_is_symmetric_and_positive(self):
        kernel = build_transient_kernel(13, rho=0.8, kernel_mode="hybrid", power_alpha=1.2, kernel_mix=0.6)
        self.assertEqual(kernel.shape, (13, 13))
        self.assertTrue(np.all(kernel > 0))
        self.assertTrue(np.allclose(kernel, kernel.T))
        self.assertTrue(np.all(np.diag(kernel) <= 1.0 + 1e-12))

    def test_participation_cap_adapts_but_is_bounded(self):
        cap_high_vol = effective_participation_cap(0.25, liq_z_day=0.0, vol_z_day=2.0, adaptive=True)
        cap_high_liq = effective_participation_cap(0.25, liq_z_day=2.0, vol_z_day=0.0, adaptive=True)
        self.assertTrue(0.05 <= cap_high_vol <= 0.35)
        self.assertTrue(0.05 <= cap_high_liq <= 0.35)
        self.assertLess(cap_high_vol, cap_high_liq)

    def test_new_policy_schedules_conserve_quantity(self):
        Q = -90000.0
        n_slices = 13
        V_day = 2.1e7
        max_participation = 0.22
        schedule_pov = pov_schedule(Q, V_day, max_participation, n_slices)
        schedule_front = front_loaded_schedule(Q, n_slices, decay=0.3)
        self.assertAlmostEqual(float(np.sum(schedule_front)), Q, delta=1e-8)
        self.assertAlmostEqual(float(np.sum(schedule_pov)), Q, delta=1e-8)
        self.assertTrue(np.all(np.sign(schedule_front[np.abs(schedule_front) > 0]) == np.sign(Q)))
        self.assertTrue(np.all(np.sign(schedule_pov[np.abs(schedule_pov) > 0]) == np.sign(Q)))


if __name__ == "__main__":
    unittest.main()
