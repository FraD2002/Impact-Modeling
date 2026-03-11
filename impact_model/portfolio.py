import numpy as np
import pandas as pd

from impact_model.execution import optimize_portfolio_schedule


def build_latest_portfolio_schedule(prepared, params):
    unique_days = np.unique(prepared.day_idx)
    latest_day = int(unique_days[-1])
    x_latest = prepared.x_matrix[:, latest_day]
    V_latest = prepared.V_matrix[:, latest_day]
    sigma_latest = prepared.sigma_matrix[:, latest_day]
    liq_latest = np.log(np.maximum(V_latest, 1e-12))
    liq_z_latest = (liq_latest - prepared.feature_context["liq_mean"]) / prepared.feature_context["liq_std"]
    vol_z_latest = (sigma_latest - prepared.feature_context["vol_mean"]) / prepared.feature_context["vol_std"]
    Q_latest = np.clip(x_latest, -0.2 * V_latest, 0.2 * V_latest)
    valid_latest = np.isfinite(Q_latest) & np.isfinite(V_latest) & np.isfinite(sigma_latest) & (V_latest > 0) & (sigma_latest > 0)
    idx_latest = np.where(valid_latest)[0]

    if idx_latest.size < 2:
        return pd.DataFrame(columns=["asset_idx", "slice", "quantity"])

    ranked = idx_latest[np.argsort(np.abs(Q_latest[idx_latest]))[::-1]]
    selected = ranked[: min(5, ranked.size)]
    Y_selected = prepared.y_mid_df.to_numpy(dtype=float)[selected]
    Y_selected = np.where(np.isfinite(Y_selected), Y_selected, np.nan)
    row_means = np.nanmean(Y_selected, axis=1, keepdims=True)
    Y_selected = np.where(np.isfinite(Y_selected), Y_selected, row_means)
    cov_matrix = np.cov(Y_selected)
    corr_matrix = np.corrcoef(Y_selected)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    cross_matrix = 0.05 * corr_matrix
    np.fill_diagonal(cross_matrix, 0.0)
    Q_vec = Q_latest[selected]
    V_vec = V_latest[selected]
    sigma_vec = sigma_latest[selected]
    liq_vec = liq_z_latest[selected]
    vol_vec = vol_z_latest[selected]
    params_vec = np.tile(params, (selected.size, 1))
    portfolio_schedule = optimize_portfolio_schedule(
        Q_vec,
        V_vec,
        sigma_vec,
        params_vec,
        liq_vec,
        vol_vec,
        cov_matrix,
        cross_matrix,
        lambda_risk=1e-6,
        lambda_port=1e-6,
        max_participation=0.25,
        n_slices=13,
        adaptive_participation=True,
        kernel_mode="hybrid",
        power_alpha=1.25,
        kernel_mix=0.55,
        spread_penalty_scale=0.02,
    )
    portfolio_rows = []
    for i, asset_id in enumerate(selected):
        for t in range(portfolio_schedule.shape[1]):
            portfolio_rows.append(
                {
                    "asset_idx": int(asset_id),
                    "slice": int(t),
                    "quantity": float(portfolio_schedule[i, t]),
                }
            )
    return pd.DataFrame(portfolio_rows)
