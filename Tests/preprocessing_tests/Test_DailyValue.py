import unittest

import numpy as np

from Preprocessing.DailyValue import getDailyValue
from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class TestgetDailyValue(unittest.TestCase):
    @unittest.skipUnless(raw_trade_file("20070919/IBM_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_constructor(self):
        data = TAQTradesReader(str(raw_trade_file("20070919/IBM_trades.binRT")))
        tdv = np.round(getDailyValue(data) / (10**9), 4)
        expected_tdv = np.round(1.043658, 4)
        np.testing.assert_array_equal(tdv, expected_tdv)


if __name__ == "__main__":
    unittest.main()
