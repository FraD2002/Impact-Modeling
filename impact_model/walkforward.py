from dataclasses import dataclass

import numpy as np
import pandas as pd

from impact_model.calibration import (
    apply_clips,
    baseline_predict,
    classify_regime,
    clip_param_drift,
    compute_metrics,
    fit_baseline_model,
    fit_structural_model,
    get_clip_bounds,
    predict_mid_term,
)
from impact_model.execution import (
    effective_participation_cap,
    front_loaded_schedule,
    optimize_schedule_single_asset,
    pov_schedule,
    schedule_cost_components,
    twap_schedule,
    vwap_schedule,
)


@dataclass
class WalkforwardResult:
    walkforward_df: pd.DataFrame
    comparison_df: pd.DataFrame
    policy_df: pd.DataFrame
    online_df: pd.DataFrame
    oos_summary: dict
    comparison_summary: list[dict]
    policy_summary: dict


def choose_lookback_window(previous_oos_rmse, default_lookback=60, short_lookback=20, threshold=1.15):
    recalibration_trigger = False
    lookback_days = default_lookback
    if len(previous_oos_rmse) >= 5:
        recent = np.array(previous_oos_rmse[-5:], dtype=float)
        baseline_level = float(np.nanmedian(recent))
        if np.isfinite(baseline_level) and baseline_level > 0 and previous_oos_rmse[-1] > threshold * baseline_level:
            lookback_days = short_lookback
            recalibration_trigger = True
    return lookback_days, recalibration_trigger


def _joint_metrics(y_mid_true, y_term_true, y_mid_pred, y_term_pred):
    rmse_mid, mae_mid, directional_mid = compute_metrics(y_mid_true, y_mid_pred)
    rmse_term, mae_term, directional_term = compute_metrics(y_term_true, y_term_pred)
    rmse_joint, mae_joint, directional_joint = compute_metrics(
        np.concatenate([y_mid_true, y_term_true]),
        np.concatenate([y_mid_pred, y_term_pred]),
    )
    return {
        "rmse_mid": rmse_mid,
        "mae_mid": mae_mid,
        "directional_mid": directional_mid,
        "rmse_term": rmse_term,
        "mae_term": mae_term,
        "directional_term": directional_term,
        "rmse_joint": rmse_joint,
        "mae_joint": mae_joint,
        "directional_joint": directional_joint,
    }


def summarize_walkforward(walkforward_df):
    if walkforward_df.empty:
        return {
            "rmse_joint": float("nan"),
            "mae_joint": float("nan"),
            "directional_joint": float("nan"),
            "rmse_mid": float("nan"),
            "rmse_term": float("nan"),
            "evaluated_days": 0,
        }
    return {
        "rmse_joint": float(np.nanmean(walkforward_df["rmse_joint"])),
        "mae_joint": float(np.nanmean(walkforward_df["mae_joint"])),
        "directional_joint": float(np.nanmean(walkforward_df["directional_joint"])),
        "rmse_mid": float(np.nanmean(walkforward_df["rmse_mid"])),
        "rmse_term": float(np.nanmean(walkforward_df["rmse_term"])),
        "evaluated_days": int(len(walkforward_df)),
    }


def summarize_model_comparison(comparison_df):
    if comparison_df.empty:
        return []
    return (
        comparison_df.groupby("model")[["rmse_joint", "mae_joint", "directional_joint"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )


def summarize_policy_benchmarks(policy_df):
    if policy_df.empty:
        return {
            "evaluated_assets": 0,
            "policies": [],
        }
    comparison_df = policy_df[policy_df["policy"] != "optimized"].copy()
    summary_rows = []
    for policy_name, group in comparison_df.groupby("policy"):
        summary_rows.append(
            {
                "policy": policy_name,
                "avg_cost_total": float(np.nanmean(group["cost_total"])),
                "avg_saving_vs_optimized": float(np.nanmean(group["saving_vs_optimized"])),
                "optimized_win_rate": float(np.nanmean(group["optimized_wins"])),
            }
        )
    summary_rows.sort(key=lambda row: row["avg_saving_vs_optimized"], reverse=True)
    return {
        "evaluated_assets": int(policy_df[["test_day", "asset_idx"]].drop_duplicates().shape[0]),
        "policies": summary_rows,
    }


def _policy_candidates(Q_target, V_day, max_participation, n_slices):
    cap = max_participation
    return {
        "twap": twap_schedule(Q_target, n_slices),
        "vwap": vwap_schedule(Q_target, n_slices),
        "pov": pov_schedule(Q_target, V_day, cap, n_slices),
        "front_loaded": front_loaded_schedule(Q_target, n_slices, decay=0.24),
    }


def _append_policy_rows(policy_rows, test_day, asset_idx, Q_target, costs_by_policy):
    optimized_cost = costs_by_policy["optimized"]["total"]
    for policy_name, components in costs_by_policy.items():
        total_cost = float(components["total"])
        saving_vs_optimized = float(total_cost - optimized_cost)
        policy_rows.append(
            {
                "test_day": int(test_day),
                "asset_idx": int(asset_idx),
                "Q_target": float(Q_target),
                "policy": policy_name,
                "cost_total": total_cost,
                "temp": float(components["temp"]),
                "transient": float(components["transient"]),
                "perm": float(components["perm"]),
                "risk": float(components["risk"]),
                "spread": float(components.get("spread", 0.0)),
                "saving_vs_optimized": saving_vs_optimized,
                "optimized_wins": float(saving_vs_optimized >= 0.0) if policy_name != "optimized" else float("nan"),
            }
        )


def run_walkforward_analysis(prepared):
    sigma = prepared.sigma
    day_idx = prepared.day_idx
    feature_context = prepared.feature_context
    V_matrix = prepared.V_matrix
    sigma_matrix = prepared.sigma_matrix
    x_matrix = prepared.x_matrix

    x_filtered = prepared.x_filtered
    y_mid_filtered = prepared.y_mid_filtered
    y_term_filtered = prepared.y_term_filtered
    sigma_filtered = prepared.sigma_filtered
    features_filtered = prepared.features_filtered
    day_idx_filtered = prepared.day_idx_filtered

    walkforward_rows = []
    model_comparison_rows = []
    policy_rows = []
    online_rows = []
    regime_params = {}
    previous_oos_rmse = []
    unique_days = np.unique(day_idx)
    min_train_days = max(10, int(np.ceil(0.35 * unique_days.size)))

    for i in range(min_train_days, unique_days.size):
        test_day = unique_days[i]
        lookback_days, recalibration_trigger = choose_lookback_window(previous_oos_rmse)
        start_i = max(0, i - lookback_days)
        train_days = unique_days[start_i:i]
        if train_days.size < 5:
            continue
        train_mask = np.isin(day_idx_filtered, train_days)
        test_mask = day_idx_filtered == test_day
        if np.sum(train_mask) < 300 or np.sum(test_mask) < 20:
            continue

        x_train_raw = x_filtered[train_mask]
        y_mid_train_raw = y_mid_filtered[train_mask]
        y_term_train_raw = y_term_filtered[train_mask]
        sigma_train = sigma_filtered[train_mask]
        features_train = features_filtered[train_mask]
        duration_train = np.clip(np.abs(x_train_raw) / (np.quantile(np.abs(x_train_raw), 0.75) + 1e-12), 0.0, 5.0)

        x_test_raw = x_filtered[test_mask]
        y_mid_test_raw = y_mid_filtered[test_mask]
        y_term_test_raw = y_term_filtered[test_mask]
        sigma_test = sigma_filtered[test_mask]
        features_test = features_filtered[test_mask]
        duration_test = np.clip(np.abs(x_test_raw) / (np.quantile(np.abs(x_train_raw), 0.75) + 1e-12), 0.0, 5.0)

        x_train_bounds = get_clip_bounds(x_train_raw)
        y_mid_train_bounds = get_clip_bounds(y_mid_train_raw)
        y_term_train_bounds = get_clip_bounds(y_term_train_raw)

        x_train = apply_clips(x_train_raw, x_train_bounds)
        y_mid_train = apply_clips(y_mid_train_raw, y_mid_train_bounds)
        y_term_train = apply_clips(y_term_train_raw, y_term_train_bounds)
        x_test = apply_clips(x_test_raw, x_train_bounds)
        y_mid_test = apply_clips(y_mid_test_raw, y_mid_train_bounds)
        y_term_test = apply_clips(y_term_test_raw, y_term_train_bounds)

        train_sigma_per_day = np.array([np.median(sigma[day_idx == d]) for d in train_days])
        test_sigma_value = float(np.median(sigma[test_mask]))
        regime = classify_regime(train_sigma_per_day, test_sigma_value)
        prior = regime_params.get(regime)
        smoothing_weight = 2.0 if prior is not None else 0.0

        try:
            params_day = fit_structural_model(
                x_train,
                y_mid_train,
                y_term_train,
                sigma_train,
                features_train,
                duration_train,
                prior_params=prior,
                smoothing_weight=smoothing_weight,
            )
            if prior is not None:
                params_day = clip_param_drift(params_day, prior)
        except RuntimeError:
            continue

        if prior is None:
            regime_params[regime] = params_day
        else:
            regime_params[regime] = 0.85 * prior + 0.15 * params_day

        baseline_day = fit_baseline_model(x_train, y_mid_train, y_term_train, sigma_train)

        mid_pred_day, term_pred_day = predict_mid_term(params_day, x_test, sigma_test, features_test, duration_test)
        base_mid_day, base_term_day = baseline_predict(baseline_day, x_test, sigma_test)
        zero_mid_day = np.zeros_like(y_mid_test)
        zero_term_day = np.zeros_like(y_term_test)

        structural_metrics = _joint_metrics(y_mid_test, y_term_test, mid_pred_day, term_pred_day)
        baseline_metrics = _joint_metrics(y_mid_test, y_term_test, base_mid_day, base_term_day)
        zero_metrics = _joint_metrics(y_mid_test, y_term_test, zero_mid_day, zero_term_day)

        previous_oos_rmse.append(structural_metrics["rmse_joint"])

        walkforward_rows.append(
            {
                "test_day": int(test_day),
                "lookback_days": int(lookback_days),
                "recalibration_trigger": bool(recalibration_trigger),
                "regime": regime,
                "n_train": int(np.sum(train_mask)),
                "n_test": int(np.sum(test_mask)),
                "eta_buy": float(params_day[0]),
                "eta_sell": float(params_day[1]),
                "beta": float(params_day[2]),
                "phi_buy": float(params_day[3]),
                "phi_sell": float(params_day[4]),
                "delta": float(params_day[5]),
                "kappa": float(params_day[6]),
                "rho": float(params_day[7]),
                **structural_metrics,
            }
        )

        for model_name, metric_values in [
            ("structural", structural_metrics),
            ("baseline", baseline_metrics),
            ("zero", zero_metrics),
        ]:
            model_comparison_rows.append(
                {
                    "test_day": int(test_day),
                    "model": model_name,
                    **metric_values,
                }
            )

        day_col = int(test_day)
        x_day_raw = x_matrix[:, day_col]
        V_day_raw = V_matrix[:, day_col]
        sigma_day_raw = sigma_matrix[:, day_col]
        liq_day_raw = np.log(np.maximum(V_day_raw, 1e-12))
        liq_z_day_raw = (liq_day_raw - feature_context["liq_mean"]) / feature_context["liq_std"]
        vol_z_day_raw = (sigma_day_raw - feature_context["vol_mean"]) / feature_context["vol_std"]
        valid_day = np.isfinite(x_day_raw) & np.isfinite(V_day_raw) & np.isfinite(sigma_day_raw) & (V_day_raw > 0) & (sigma_day_raw > 0)
        if np.any(valid_day):
            Q_target_vec = np.clip(x_day_raw, -0.2 * V_day_raw, 0.2 * V_day_raw)
            Q_target_vec[~valid_day] = 0.0
            selected_assets = np.where(np.abs(Q_target_vec) > 0)[0]
            if selected_assets.size > 0:
                ranked = selected_assets[np.argsort(np.abs(Q_target_vec[selected_assets]))[::-1]]
                selected_assets = ranked[: min(3, ranked.size)]
                for asset_idx in selected_assets:
                    Q_target = float(Q_target_vec[asset_idx])
                    V_day = float(V_day_raw[asset_idx])
                    sigma_day = float(sigma_day_raw[asset_idx])
                    liq_z_day = float(liq_z_day_raw[asset_idx])
                    vol_z_day = float(vol_z_day_raw[asset_idx])
                    adaptive_cap = effective_participation_cap(0.25, liq_z_day, vol_z_day, adaptive=True)

                    optimized_schedule, optimized_cost = optimize_schedule_single_asset(
                        Q_target,
                        V_day,
                        sigma_day,
                        params_day,
                        liq_z_day,
                        vol_z_day,
                        lambda_risk=1e-6,
                        max_participation=0.25,
                        n_slices=13,
                        adaptive_participation=True,
                        kernel_mode="hybrid",
                        power_alpha=1.25,
                        kernel_mix=0.55,
                        spread_penalty_scale=0.02,
                    )
                    Q_executed = float(np.sign(Q_target) * np.sum(np.abs(optimized_schedule)))
                    if np.abs(Q_executed) <= 1e-12:
                        continue
                    costs_by_policy = {"optimized": optimized_cost}
                    for policy_name, schedule in _policy_candidates(Q_executed, V_day, adaptive_cap, 13).items():
                        costs_by_policy[policy_name] = schedule_cost_components(
                            np.abs(schedule),
                            Q_executed,
                            V_day,
                            sigma_day,
                            params_day,
                            liq_z_day,
                            vol_z_day,
                            1e-6,
                            kernel_mode="hybrid",
                            power_alpha=1.25,
                            kernel_mix=0.55,
                            spread_penalty_scale=0.02,
                        )
                    _append_policy_rows(policy_rows, test_day, asset_idx, Q_executed, costs_by_policy)

        online_rows.append(
            {
                "test_day": int(test_day),
                "lookback_days": int(lookback_days),
                "recalibration_trigger": bool(recalibration_trigger),
                "regime": regime,
                "oos_rmse_joint": structural_metrics["rmse_joint"],
            }
        )

    walkforward_df = pd.DataFrame(walkforward_rows)
    comparison_df = pd.DataFrame(model_comparison_rows)
    policy_df = pd.DataFrame(policy_rows)
    online_df = pd.DataFrame(online_rows)

    return WalkforwardResult(
        walkforward_df=walkforward_df,
        comparison_df=comparison_df,
        policy_df=policy_df,
        online_df=online_df,
        oos_summary=summarize_walkforward(walkforward_df),
        comparison_summary=summarize_model_comparison(comparison_df),
        policy_summary=summarize_policy_benchmarks(policy_df),
    )
