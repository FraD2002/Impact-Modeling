import unittest

import numpy as np

from Preprocessing.Imbalance import getImbalance
from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class TestImbalance(unittest.TestCase):
    @unittest.skipUnless(raw_trade_file("20070919/IBM_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_constructor(self):
        startTS = 18 * 60 * 60 * 1000 // 2
        endTS = startTS + (13 * 60 * 60 * 1000 // 2)
        data = TAQTradesReader(str(raw_trade_file("20070919/IBM_trades.binRT")))
        returns = round(getImbalance(data, startTS, endTS), 2)
        expected_returns = round(-45224126.443963, 2)
        np.testing.assert_array_equal(returns, expected_returns)


if __name__ == "__main__":
    unittest.main()
