import unittest

from impactUtils.VWAP import VWAP
from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class Test_VWAP(unittest.TestCase):
    @unittest.skipUnless(raw_trade_file("20070919/IBM_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_vwap(self):
        start930 = 19 * 60 * 60 * 1000 / 2
        end4 = 16 * 60 * 60 * 1000
        vwap = VWAP(TAQTradesReader(str(raw_trade_file("20070919/IBM_trades.binRT"))), start930, end4)
        self.assertEqual(vwap.getN(), 36913)
        self.assertAlmostEqual(vwap.getVWAP(), 116.468791, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
