import math

class LastPriceBuckets(object):
    DEFAULT_START_TS = 19 * 60 * 60 * 1000 / 2
    DEFAULT_END_TS = 16 * 60 * 60 * 1000

    def __init__(
            self,
            data,  # Must implement getPrice(i), getTimestamp(i), and getN()
            numBuckets,
            startTS,  # eg 930AM = 19 * 60 * 60 * 1000 / 2
            endTS  # eg. 4PM = 16 * 60 * 60 * 10000
    ):
        if numBuckets <= 0:
            raise ValueError("numBuckets must be positive.")

        if startTS is None:
            startTS = type(self).DEFAULT_START_TS
        if endTS is None:
            endTS = type(self).DEFAULT_END_TS
        if endTS <= startTS:
            raise ValueError("endTS must be greater than startTS.")

        self._startTS = startTS
        self._endTS = endTS
        self._numBuckets = numBuckets
        bucketLen = (endTS - startTS) / numBuckets

        # Initialize timestamp and price lists
        self._timestamps = [None] * numBuckets
        self._prices = [None] * numBuckets

        nRecs = data.getN()
        for startI in range(0, nRecs):
            timestamp = data.getTimestamp(startI)
            # Are we past the end of good data?
            if timestamp >= endTS:
                # Yes, we are past the end of good data
                # Stop computing data buckets
                break
            # Are we still iterating over data before the specified start?
            if timestamp < startTS:
                # Yes, we have to skip this data
                continue
            iBucket = int(math.floor((timestamp - startTS) / bucketLen))
            if iBucket >= numBuckets:
                iBucket = numBuckets - 1
            self._timestamps[iBucket] = timestamp
            self._prices[iBucket] = data.getPrice(startI)

    def _validate_index(self, index):
        if index < 0:
            index += self._numBuckets
        if index < 0 or index >= self._numBuckets:
            raise IndexError(f"Bucket index {index} out of range.")
        return index

    def getPrice(self, index):
        return self._prices[self._validate_index(index)]

    def getTimestamp(self, index):
        return self._timestamps[self._validate_index(index)]

    def getN(self):
        return len(self._prices)
