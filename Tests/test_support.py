from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRADES_DIR = PROJECT_ROOT / "trades"
QUOTES_DIR = PROJECT_ROOT / "quotes"


def has_raw_data():
    return TRADES_DIR.is_dir() and QUOTES_DIR.is_dir()


def raw_trade_file(relative_path):
    return TRADES_DIR / relative_path


def raw_quote_file(relative_path):
    return QUOTES_DIR / relative_path
