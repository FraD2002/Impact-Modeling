import sys
import unittest
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    tests_dir = project_root / "Tests"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern="Test_*.py", top_level_dir=str(project_root))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    num_failures = len(result.failures)
    num_errors = len(result.errors)
    num_skipped = len(result.skipped)
    num_tests = result.testsRun

    print(f"Tests run: {num_tests}")
    print(f"Failures: {num_failures}")
    print(f"Errors: {num_errors}")
    print(f"Skipped: {num_skipped}")

    if num_failures or num_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
