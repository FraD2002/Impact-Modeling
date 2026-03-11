import math

from impactUtils.FirstPriceBuckets import FirstPriceBuckets
from impactUtils.LastPriceBuckets import LastPriceBuckets


# This class will be used to build return buckets,
# e.g. 2 minute returns
class ReturnBuckets(object):
    DEFAULT_START_TS = 19 * 60 * 60 * 1000 / 2
    DEFAULT_END_TS = 16 * 60 * 60 * 1000

    def __init__(
        self,
        data,  # An object that implements getTimestamp(i), getPrice(i), getN()
        startTS,  # In milliseconds from midnight
        endTS,  # In milliseconds from midnight
        numBuckets  # Desired number of return buckets
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

        self._startTimestamps = [None] * numBuckets
        self._endTimestamps = [None] * numBuckets
        self._returns = [None] * numBuckets

        firstPriceBuckets = FirstPriceBuckets(data, numBuckets, self._startTS, self._endTS)
        lastPriceBuckets = LastPriceBuckets(data, numBuckets, self._startTS, self._endTS)
        for i in range(0, firstPriceBuckets.getN()):
            startTimestamp = firstPriceBuckets.getTimestamp(i)
            endTimestamp = lastPriceBuckets.getTimestamp(i)
            startPrice = firstPriceBuckets.getPrice(i)
            endPrice = lastPriceBuckets.getPrice(i)
            if startTimestamp is None or endTimestamp is None or startPrice is None or endPrice is None:
                continue
            startPrice = float(startPrice)
            endPrice = float(endPrice)
            if not math.isfinite(startPrice) or not math.isfinite(endPrice) or startPrice == 0:
                continue
            self._startTimestamps[i] = startTimestamp
            self._endTimestamps[i] = endTimestamp
            self._returns[i] = (endPrice / startPrice) - 1.0

    def _validate_index(self, index):
        if index < 0:
            index += self._numBuckets
        if index < 0 or index >= self._numBuckets:
            raise IndexError(f"Bucket index {index} out of range.")
        return index

    # Get start time stamp of bucket specified by
    # index.
    def getStartTimestamp(self, index):
        return self._startTimestamps[self._validate_index(index)]

    # Get end time stamp of bucket specified by
    # index.
    def getEndTimestamp(self, index):
        return self._endTimestamps[self._validate_index(index)]

    # Get the return of bucket specified by
    # index.
    def getReturn(self, index):
        return self._returns[self._validate_index(index)]

    # Get number of returns.
    def getN(self):
        return len(self._startTimestamps)
