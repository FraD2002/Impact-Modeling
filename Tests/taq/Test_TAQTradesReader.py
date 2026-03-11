import unittest

from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class Test_TAQTradesReader(unittest.TestCase):
    @unittest.skipUnless(raw_trade_file("20070822/AAI_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_reader_values(self):
        reader = TAQTradesReader(str(raw_trade_file("20070822/AAI_trades.binRT")))
        values = [
            reader.getN(),
            reader.getSecsFromEpocToMidn(),
            reader.getMillisFromMidn(0),
            reader.getSize(0),
            reader.getPrice(0),
        ]
        expected = [25367, 1190260800, 34210000, 76600, 116.2699966430664]
        self.assertEqual(values, expected)


if __name__ == "__main__":
    unittest.main()
