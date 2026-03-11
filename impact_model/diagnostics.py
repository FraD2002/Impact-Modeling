import numpy as np
import pandas as pd

from impact_model.calibration import compute_metrics, fit_structural_model


def run_bootstrap(prepared, bootstrap_iterations, seed=42):
    rng = np.random.default_rng(seed)
    bootstrap_params = []
    for _ in range(bootstrap_iterations):
        sample_idx = rng.integers(0, prepared.x.size, size=prepared.x.size)
        x_boot = prepared.x[sample_idx]
        y_mid_boot = prepared.y_mid[sample_idx]
        y_term_boot = prepared.y_term[sample_idx]
        sigma_boot = prepared.sigma[sample_idx]
        features_boot = prepared.features[sample_idx]
        duration_boot = prepared.duration[sample_idx]
        try:
            params_boot = fit_structural_model(x_boot, y_mid_boot, y_term_boot, sigma_boot, features_boot, duration_boot)
            bootstrap_params.append(params_boot)
        except RuntimeError:
            continue

    if len(bootstrap_params) >= 10:
        bootstrap_arr = np.array(bootstrap_params)
        ci_lower = np.quantile(bootstrap_arr, 0.025, axis=0)
        ci_upper = np.quantile(bootstrap_arr, 0.975, axis=0)
    else:
        ci_lower = np.full(14, np.nan)
        ci_upper = np.full(14, np.nan)
    return ci_lower, ci_upper, bootstrap_params


def compute_regime_performance(prepared, mid_pred, term_pred):
    sigma = prepared.sigma
    y_mid = prepared.y_mid
    y_term = prepared.y_term
    median_sigma = float(np.median(sigma))
    low_vol_mask = sigma <= median_sigma
    high_vol_mask = sigma > median_sigma
    low_rmse, low_mae, low_directional_error = compute_metrics(
        np.concatenate([y_mid[low_vol_mask], y_term[low_vol_mask]]),
        np.concatenate([mid_pred[low_vol_mask], term_pred[low_vol_mask]]),
    )
    high_rmse, high_mae, high_directional_error = compute_metrics(
        np.concatenate([y_mid[high_vol_mask], y_term[high_vol_mask]]),
        np.concatenate([mid_pred[high_vol_mask], term_pred[high_vol_mask]]),
    )
    return {
        "low_vol": {
            "rmse_joint": low_rmse,
            "mae_joint": low_mae,
            "directional_joint": low_directional_error,
            "n": int(np.sum(low_vol_mask)),
        },
        "high_vol": {
            "rmse_joint": high_rmse,
            "mae_joint": high_mae,
            "directional_joint": high_directional_error,
            "n": int(np.sum(high_vol_mask)),
        },
    }


def build_residual_diagnostics(prepared, mid_pred, term_pred):
    return pd.DataFrame(
        {
            "asset_idx": prepared.asset_idx,
            "day_idx": prepared.day_idx,
            "x": prepared.x,
            "sigma": prepared.sigma,
            "y_mid": prepared.y_mid,
            "y_term": prepared.y_term,
            "mid_pred": mid_pred,
            "term_pred": term_pred,
            "mid_residual": prepared.y_mid - mid_pred,
            "term_residual": prepared.y_term - term_pred,
        }
    )
