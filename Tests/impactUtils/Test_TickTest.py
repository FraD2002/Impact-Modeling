import unittest

from impactUtils.TickTest import TickTest
from taq.TAQTradesReader import TAQTradesReader
from Tests.test_support import raw_trade_file


class Test_TickTest(unittest.TestCase):
    def test_classify(self):
        tickTest = TickTest()
        self.assertEqual(tickTest.classify(100), 0)
        self.assertEqual(tickTest.classify(100), 0)
        self.assertEqual(tickTest.classify(101), 1)
        self.assertEqual(tickTest.classify(101), 1)
        self.assertEqual(tickTest.classify(102), 1)
        self.assertEqual(tickTest.classify(101), -1)
        self.assertEqual(tickTest.classify(101), -1)
        self.assertEqual(tickTest.classify(102), 1)

    @unittest.skipUnless(raw_trade_file("20070919/IBM_trades.binRT").is_file(), "Raw TAQ trade file not available.")
    def test_classify_all(self):
        data = TAQTradesReader(str(raw_trade_file("20070919/IBM_trades.binRT")))
        tickTest = TickTest()
        startOfDay = 18 * 60 * 60 * 1000 / 2
        endOfDay = startOfDay + (13 * 60 * 60 * 1000 / 2)
        classifications = tickTest.classifyAll(data, startOfDay, endOfDay)
        expected = [
            (34216000, 116.9000015258789, 0),
            (34216000, 116.9000015258789, 0),
            (34216000, 116.94999694824219, 1),
            (34216000, 116.98999786376953, 1),
            (34216000, 117.19000244140625, 1),
            (34216000, 117.18000030517578, -1),
            (34219000, 117.0199966430664, -1),
            (34219000, 117.02999877929688, 1),
            (34219000, 116.7699966430664, -1),
            (34219000, 116.7699966430664, -1),
        ]
        self.assertEqual(classifications[0:10], expected)


if __name__ == "__main__":
    unittest.main()
