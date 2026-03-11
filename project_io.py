from pathlib import Path

import pandas as pd


def ensure_required_files(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))


def ensure_required_directories(paths):
    missing = [str(path) for path in paths if not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError("Missing required directories: " + ", ".join(missing))


def read_required_csvs(base_dir, file_names, index_col=None):
    base_dir = Path(base_dir)
    paths = [base_dir / file_name for file_name in file_names]
    ensure_required_files(paths)
    frames = {}
    for path in paths:
        frames[path.stem] = pd.read_csv(path, index_col=index_col)
    return frames


def write_csv_frames(output_dir, frames, index_label=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index_label=index_label)


def read_nonempty_lines(file_path):
    file_path = Path(file_path)
    ensure_required_files([file_path])
    with file_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        raise ValueError(f"{file_path} does not contain any non-empty rows.")
    return lines


def drop_columns_if_present(frame, columns):
    return frame.drop(columns=list(columns), errors="ignore")
