import unittest

from impact_model.evaluation import choose_lookback_window


class Test_OnlineRecalibration(unittest.TestCase):
    def test_triggered_recalibration_uses_short_lookback(self):
        previous_oos_rmse = [1.0, 1.05, 0.98, 1.02, 1.4]
        lookback_days, recalibration_trigger = choose_lookback_window(previous_oos_rmse)
        self.assertEqual(lookback_days, 20)
        self.assertTrue(recalibration_trigger)

    def test_default_lookback_without_trigger(self):
        previous_oos_rmse = [1.0, 1.03, 0.99, 1.01, 1.08]
        lookback_days, recalibration_trigger = choose_lookback_window(previous_oos_rmse)
        self.assertEqual(lookback_days, 60)
        self.assertFalse(recalibration_trigger)

    def test_default_lookback_with_insufficient_history(self):
        previous_oos_rmse = [1.0, 1.2, 1.1]
        lookback_days, recalibration_trigger = choose_lookback_window(previous_oos_rmse)
        self.assertEqual(lookback_days, 60)
        self.assertFalse(recalibration_trigger)


if __name__ == "__main__":
    unittest.main()
