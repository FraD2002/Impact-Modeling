import numpy as np
from impactUtils.ReturnBuckets import ReturnBuckets


def getMidQuoteReturns(quotes, startTS, endTS, numBuckets):
    if numBuckets <= 0:
        raise ValueError("numBuckets must be positive.")

    returnBuckets = ReturnBuckets(quotes, startTS, endTS, numBuckets)
    quoteOneDayReturns = np.zeros(numBuckets, dtype=float)
    max_len = min(numBuckets, returnBuckets.getN())

    for i in range(max_len):
        return_val = returnBuckets.getReturn(i)
        if return_val is not None and np.isfinite(return_val):
            quoteOneDayReturns[i] = float(return_val)

    return quoteOneDayReturns
