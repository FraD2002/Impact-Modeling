import gzip
import os
import struct


class TAQTradesReader:
    def __init__(self, filePathName):
        self._filePathName = filePathName
        if not os.path.isfile(self._filePathName):
            raise FileNotFoundError(f"Trades file not found: {self._filePathName}")

        try:
            with gzip.open(self._filePathName, "rb") as file:
                file_content = file.read()
        except OSError as exc:
            raise ValueError(f"Unable to read compressed trades file: {self._filePathName}") from exc

        if len(file_content) < 8:
            raise ValueError(f"Trades file is too short to contain a valid header: {self._filePathName}")

        try:
            self._header = struct.unpack_from(">2i", file_content, 0)
        except struct.error as exc:
            raise ValueError(f"Invalid trades file header: {self._filePathName}") from exc

        self._secs_from_epoc_to_midn = self._header[0]
        self._n = self._header[1]
        if self._n < 0:
            raise ValueError(f"Invalid negative record count in trades file: {self._filePathName}")

        expected_bytes = 8 + (12 * self._n)
        if len(file_content) < expected_bytes:
            raise ValueError(
                f"Corrupted trades file {self._filePathName}: expected at least {expected_bytes} bytes, got {len(file_content)}."
            )

        offset = 8
        self._ts = struct.unpack_from(f">{self._n}i", file_content, offset)
        offset += 4 * self._n
        self._s = struct.unpack_from(f">{self._n}i", file_content, offset)
        offset += 4 * self._n
        self._p = struct.unpack_from(f">{self._n}f", file_content, offset)

    def _normalize_index(self, index):
        if index < 0:
            index += self._n
        if index < 0 or index >= self._n:
            raise IndexError(f"Trade index {index} out of range [0, {self._n - 1}] in {self._filePathName}")
        return index

    def getN(self):
        return self._n

    def getSecsFromEpocToMidn(self):
        return self._secs_from_epoc_to_midn

    def getPrice(self, index):
        index = self._normalize_index(index)
        return self._p[index]

    def getMillisFromMidn(self, index):
        index = self._normalize_index(index)
        return self._ts[index]

    def getTimestamp(self, index):
        return self.getMillisFromMidn(index)

    def getSize(self, index):
        index = self._normalize_index(index)
        return self._s[index]

    def rewrite(self, filePathName, tickerId):
        if tickerId < 0 or tickerId > 65535:
            raise ValueError("tickerId must be in [0, 65535] for binary packing.")

        packer = struct.Struct(">QHIf")
        baseTS = self.getSecsFromEpocToMidn() * 1000
        with gzip.open(filePathName, "wb") as out:
            for i in range(self.getN()):
                ts = baseTS + self.getMillisFromMidn(i)
                out.write(packer.pack(ts, tickerId, self.getSize(i), self.getPrice(i)))
