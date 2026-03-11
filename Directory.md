# Project Layout

## Core Pipelines
- `Preprocessing.py`: builds raw feature tables from TAQ quote/trade binaries into `Data/`.
- `Inputs.py`: transforms `Data/` into calibrated model inputs under `Input/`.
- `impactModel.py`: execution entrypoint for the impact modeling framework.

## Modeling Package
- `impact_model/calibration.py`: constrained robust calibration and prediction functions.
- `impact_model/execution.py`: state-aware execution cost model, transient kernels, and single/portfolio optimizers.
- `impact_model/data_pipeline.py`: input loading, filtering, clipping, and matrix preparation.
- `impact_model/walkforward.py`: walk-forward OOS loop, recalibration policy, benchmark policy evaluation.
- `impact_model/diagnostics.py`: bootstrap confidence intervals and residual/regime diagnostics.
- `impact_model/portfolio.py`: latest-day cross-impact portfolio schedule generation.
- `impact_model/evaluation.py`: compatibility export layer for evaluation functions.
- `impact_model/runner.py`: orchestration for model fit, evaluation, and artifact generation.

## Data and Artifacts
- `Data/`: preprocessing outputs.
- `Input/`: model-ready daily inputs.
- `Output/`: generated diagnostics and execution artifacts from `impactModel.py`.
- `quotes/`, `trades/`: raw TAQ binary data directories (optional for full preprocessing/tests).

## Tests
- `Tests/impactUtils/`: impact utility tests.
- `Tests/preprocessing_tests/`: preprocessing function tests.
- `Tests/taq/`: TAQ reader tests.
- `Tests/model/`: synthetic tests for execution, cross-impact, and online recalibration paths.
- `Tests/model/Test_AdvancedExecutionPolicies.py`: transient-kernel, adaptive-cap, and advanced policy schedule tests.
- `Tests/Test_regression.py`: regression bootstrap test.

## Utilities
- `project_io.py`: shared file/directory and CSV IO helpers used across pipeline stages.
- `runTests.py`: adaptive test runner (`all tests` with raw data, otherwise regression + model synthetic tests).
