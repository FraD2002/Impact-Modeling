import unittest

import numpy as np

from Preprocessing.MidQuoteReturns import getMidQuoteReturns
from taq.TAQQuotesReader import TAQQuotesReader
from Tests.test_support import raw_quote_file


class TestGetMidQuoteReturns(unittest.TestCase):
    @unittest.skipUnless(raw_quote_file("20070919/IBM_quotes.binRQ").is_file(), "Raw TAQ quote file not available.")
    def test_constructor(self):
        startTS = 18 * 60 * 60 * 1000 // 2
        endTS = startTS + (13 * 60 * 60 * 1000 // 2)
        numBuckets = 4
        data = TAQQuotesReader(str(raw_quote_file("20070919/IBM_quotes.binRQ")))
        returns = np.round(getMidQuoteReturns(data, startTS, endTS, numBuckets), 4)
        expected_returns = np.round(np.array([-0.005313, 0.003575, 0.001974, 0]), 4)
        np.testing.assert_array_equal(returns, expected_returns)


if __name__ == "__main__":
    unittest.main()
