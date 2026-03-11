import math


# Implementation of tick test for classifying
# trades as either buyer (1) or seller (-1) initiated.
# Starting classification is undetermined (0).
class TickTest(object):

    # We need a tolerance to determine if price
    # has changed
    TOLERANCE = 0.00001

    def __init__(self):
        self.side = 0
        self.prevPrice = 0

    def classify(self, newPrice):
        if not math.isfinite(newPrice):
            raise ValueError("newPrice must be finite.")
        if self.prevPrice != 0:
            if newPrice > (self.prevPrice + type(self).TOLERANCE):
                self.side = 1
            elif newPrice < (self.prevPrice - type(self).TOLERANCE):
                self.side = -1
        self.prevPrice = newPrice
        return self.side

    def classifyAll(self, data, startTimestamp, endTimestamp):
        if endTimestamp <= startTimestamp:
            raise ValueError("endTimestamp must be greater than startTimestamp.")

        classifications = []
        for i in range(0, data.getN()):
            if data.getTimestamp(i) < startTimestamp:
                continue
            if data.getTimestamp(i) >= endTimestamp:
                break
            price = data.getPrice(i)
            classifications.append((data.getTimestamp(i), price, self.classify(price)))
        return classifications
