import os
from pathlib import Path


class MyDirectories:
    def __init__(self, root_path):
        self.root_path = root_path

    @staticmethod
    def _repo_root():
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def getQuotesDir():
        env_override = os.environ.get("TAQ_QUOTES_DIR")
        if env_override:
            return str(Path(env_override).resolve())
        return str(MyDirectories._repo_root() / "quotes")

    @staticmethod
    def getTradesDir():
        env_override = os.environ.get("TAQ_TRADES_DIR")
        if env_override:
            return str(Path(env_override).resolve())
        return str(MyDirectories._repo_root() / "trades")
