import unittest

from taq.TAQQuotesReader import TAQQuotesReader
from Tests.test_support import raw_quote_file


class Test_TAQQuotesReader(unittest.TestCase):
    @unittest.skipUnless(raw_quote_file("20070822/AAI_quotes.binRQ").is_file(), "Raw TAQ quote file not available.")
    def test_reader_values(self):
        reader = TAQQuotesReader(str(raw_quote_file("20070822/AAI_quotes.binRQ")))
        values = [
            reader.getN(),
            reader.getSecsFromEpocToMidn(),
            reader.getMillisFromMidn(0),
            reader.getAskSize(0),
            reader.getAskPrice(0),
            reader.getBidSize(0),
            reader.getBidPrice(0),
        ]
        expected = [10476, 1187755200, 34252000, 54, 10.380000114440918, 20, 10.300000190734863]
        self.assertEqual(values, expected)


if __name__ == "__main__":
    unittest.main()
