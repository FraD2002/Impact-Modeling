import unittest

from impactUtils.LastPriceBuckets import LastPriceBuckets


class StubTrades:
    def __init__(self):
        self._data = [
            [10 * 60 * 60 * 1000, 59.30],
            [15 * 60 * 60 * 1000, 63.00],
            [1 + 15 * 60 * 60 * 1000, 63.52],
        ]

    def getN(self):
        return len(self._data)

    def getTimestamp(self, iRec):
        return self._data[iRec][0]

    def getPrice(self, iRec):
        return self._data[iRec][1]


class Test_LastPriceBuckets(unittest.TestCase):
    def test_constructor(self):
        fpb = LastPriceBuckets(StubTrades(), 2, None, None)
        self.assertEqual(fpb.getN(), 2)
        self.assertEqual(fpb.getTimestamp(0), 10 * 60 * 60 * 1000)
        self.assertEqual(fpb.getTimestamp(1), 1 + 15 * 60 * 60 * 1000)
        self.assertAlmostEqual(fpb.getPrice(0), 59.30, delta=1e-4)
        self.assertAlmostEqual(fpb.getPrice(1), 63.52, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
