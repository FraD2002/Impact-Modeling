import unittest

from Preprocessing.ArrivalPrice import getArrivalPrice


class StubTAQTradesReader:
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


class TestArrivalPrice(unittest.TestCase):
    def testConstructor(self):
        startTS = None
        endTS = None
        numBuckets = 2
        dataReader = StubTAQTradesReader()
        arrivalPrice = getArrivalPrice(dataReader, startTS, endTS, numBuckets)
        self.assertAlmostEqual(arrivalPrice, 59.30, delta=0.0001)


if __name__ == "__main__":
    unittest.main()
