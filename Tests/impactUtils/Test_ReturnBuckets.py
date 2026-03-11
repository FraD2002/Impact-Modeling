import unittest

from impactUtils.ReturnBuckets import ReturnBuckets
from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class Test_ReturnBuckets(unittest.TestCase):
    @unittest.skipUnless(raw_trade_file("20070919/IBM_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_returns(self):
        startTS = 18 * 60 * 60 * 1000 / 2
        endTS = 16 * 60 * 60 * 1000
        numBuckets = 2
        data = TAQTradesReader(str(raw_trade_file("20070919/IBM_trades.binRT")))
        returnBuckets = ReturnBuckets(data, startTS, endTS, numBuckets)

        self.assertEqual(returnBuckets.getN(), 2)
        self.assertAlmostEqual(returnBuckets.getReturn(0), -0.0035073024530167807, delta=1e-12)
        self.assertAlmostEqual(returnBuckets.getReturn(1), 0.001459211750603595, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
