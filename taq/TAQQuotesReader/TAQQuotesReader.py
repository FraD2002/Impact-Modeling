import gzip
import os
import struct


class TAQQuotesReader:
    def __init__(self, filePathName):
        self._filePathName = filePathName
        if not os.path.isfile(self._filePathName):
            raise FileNotFoundError(f"Quotes file not found: {self._filePathName}")

        try:
            with gzip.open(self._filePathName, "rb") as file:
                file_content = file.read()
        except OSError as exc:
            raise ValueError(f"Unable to read compressed quotes file: {self._filePathName}") from exc

        if len(file_content) < 8:
            raise ValueError(f"Quotes file is too short to contain a valid header: {self._filePathName}")

        try:
            self._header = struct.unpack_from(">2i", file_content, 0)
        except struct.error as exc:
            raise ValueError(f"Invalid quotes file header: {self._filePathName}") from exc

        self._secs_from_epoc_to_midn = self._header[0]
        self._n = self._header[1]
        if self._n < 0:
            raise ValueError(f"Invalid negative record count in quotes file: {self._filePathName}")

        expected_bytes = 8 + (20 * self._n)
        if len(file_content) < expected_bytes:
            raise ValueError(
                f"Corrupted quotes file {self._filePathName}: expected at least {expected_bytes} bytes, got {len(file_content)}."
            )

        offset = 8
        self._ts = struct.unpack_from(f">{self._n}i", file_content, offset)
        offset += 4 * self._n
        self._bs = struct.unpack_from(f">{self._n}i", file_content, offset)
        offset += 4 * self._n
        self._bp = struct.unpack_from(f">{self._n}f", file_content, offset)
        offset += 4 * self._n
        self._as = struct.unpack_from(f">{self._n}i", file_content, offset)
        offset += 4 * self._n
        self._ap = struct.unpack_from(f">{self._n}f", file_content, offset)

    def _normalize_index(self, index):
        if index < 0:
            index += self._n
        if index < 0 or index >= self._n:
            raise IndexError(f"Quote index {index} out of range [0, {self._n - 1}] in {self._filePathName}")
        return index

    def getTimestamp(self, index):
        return self.getMillisFromMidn(index)

    def getN(self):
        return self._n

    def getSecsFromEpocToMidn(self):
        return self._secs_from_epoc_to_midn

    def getMillisFromMidn(self, index):
        index = self._normalize_index(index)
        return self._ts[index]

    def getAskSize(self, index):
        index = self._normalize_index(index)
        return self._as[index]

    def getAskPrice(self, index):
        index = self._normalize_index(index)
        return self._ap[index]

    def getBidSize(self, index):
        index = self._normalize_index(index)
        return self._bs[index]

    def getBidPrice(self, index):
        index = self._normalize_index(index)
        return self._bp[index]

    def getPrice(self, index):
        return (self.getAskPrice(index) + self.getBidPrice(index)) / 2
