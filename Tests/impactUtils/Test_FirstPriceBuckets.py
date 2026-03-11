import unittest

from impactUtils.FirstPriceBuckets import FirstPriceBuckets
from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class Test_FirstPriceBuckets(unittest.TestCase):
    @unittest.skipUnless(raw_trade_file("20070919/IBM_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_constructor(self):
        numBuckets = 2
        data = TAQTradesReader(str(raw_trade_file("20070919/IBM_trades.binRT")))
        fpb = FirstPriceBuckets(data, numBuckets, None, None)

        self.assertEqual(fpb.getN(), 2)
        self.assertEqual([fpb.getTimestamp(0), fpb.getTimestamp(1)], [34216000, 45900000])
        self.assertAlmostEqual(fpb.getPrice(0), 116.9000015258789, delta=1e-4)
        self.assertAlmostEqual(fpb.getPrice(1), 116.43000030517578, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
