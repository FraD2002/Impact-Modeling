# Market Impact Model

## Description

This project aims to develop an impact model for algorithmic trading and quantitative strategies, inspired by Almgren et al.'s "Direct Estimation of Equity Market Impact."
References:
- [Direct Estimation of Equity Market Impact](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf)
- [Optimal Execution of Portfolio Transactions (Almgren-Chriss)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)

## Objective

The main objective is to build an impact model that can accurately estimate the market impact of trading activities using trading volume, price movements, and stock liquidity.

## Data

- Utilizes the TAQ dataset, focusing on a subset of S&P 500 stocks for liquidity.
- Uses average daily value traded instead of average daily volume traded to account for stock splits.
- Data processing involves computing various metrics such as mid-quote returns, total daily value, arrival price, value imbalance, volume-weighted average price, and terminal price.

## Methodology
1. Preprocess data to obtain daily value traded, volatility, imbalance, terminal, and arrival price.
2. Calibrate a robust constrained impact model with separate temporary and permanent components for buy and sell flows.
3. Add transient impact dynamics and state-dependent scaling based on liquidity, volatility, and seasonality.
4. Use rolling walk-forward recalibration with regime switching and parameter smoothing.
5. Compare out-of-sample predictive performance against baseline and zero-impact benchmarks.
6. Optimize execution schedules versus TWAP and VWAP, and generate a covariance-aware multi-asset cross-impact portfolio schedule.

## Architecture
- `impact_model/calibration.py`: model equations, bounds, robust calibration, and metrics.
- `impact_model/data_pipeline.py`: data loading, filtering, clipping, and feature assembly.
- `impact_model/execution.py`: state-aware execution cost model, transient kernels, and schedule optimizers.
- `impact_model/walkforward.py`: walk-forward OOS evaluation and policy benchmark generation.
- `impact_model/diagnostics.py`: bootstrap confidence intervals and residual/regime diagnostics.
- `impact_model/portfolio.py`: covariance-aware multi-asset schedule construction.
- `impact_model/evaluation.py`: compatibility export layer for evaluation APIs.
- `impact_model/runner.py`: orchestration entrypoint called by `impactModel.py`.

## Running the Code
To run the code, follow these steps: <br>

Install the required dependencies: `pip install -r requirements.txt` <br>
Run the tests: `python runTests.py` <br>
To preprocess and prepare data: `python Preprocessing.py` and `python Inputs.py` <br>
Execute the main script: `python impactModel.py` <br>

Note: You need to have the raw TAQ dataset in your `quotes` and `trades` directories to run full preprocessing and TAQ-based tests.
`runTests.py` always discovers all tests; raw-data-dependent tests are skipped automatically when TAQ files are not available.

`impactModel.py` writes diagnostics under `Output/`:
- `model_diagnostics.json`
- `walkforward_metrics.csv`
- `oos_model_comparison.csv`
- `execution_policy_comparison.csv`
- `online_recalibration.csv`
- `portfolio_schedule.csv`
- `residual_diagnostics.csv`

`execution_policy_comparison.csv` now includes `optimized`, `twap`, `vwap`, `pov`, and `front_loaded` policy costs with per-policy savings vs optimized execution.
All output CSV artifacts include reproducibility metadata columns: `run_id`, `run_timestamp_utc`, `git_commit`, `git_branch`, and `config_hash`.
`model_diagnostics.json` includes a `runtime` block with the same metadata and hashed runtime configuration.

## Results
The framework calibrates constrained impact parameters and reports in-sample, walk-forward out-of-sample, and policy benchmark metrics.
The current model uses an Almgren-Chriss style impact structure:

$$h = \eta * \sigma * (\frac{X}{VT})^\beta$$

$h$: temporary impact
$\sigma$: stock specific volatility
$X$: Daily imbalance (value)
$V$:Average daily value 
$T$: time

We discovered $\eta = 0.33244$, $\beta = 0.36127$.

The analysis of the fit of the model are in [descriptiveStats.ipynb](descriptiveStats.ipynb)
