import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from impact_model.calibration import (
    baseline_predict,
    compute_metrics,
    fit_baseline_model,
    fit_structural_model,
    predict_mid_term,
)
from impact_model.data_pipeline import load_inputs, prepare_model_data
from impact_model.evaluation import (
    build_latest_portfolio_schedule,
    build_residual_diagnostics,
    compute_regime_performance,
    run_bootstrap,
    run_walkforward_analysis,
)


def _git_value(project_root, *args):
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_runtime_metadata(project_root, runtime_config):
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    config_payload = json.dumps(runtime_config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()
    commit = _git_value(project_root, "rev-parse", "HEAD")
    branch = _git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    run_id = f"{run_timestamp_utc.replace(':', '').replace('-', '').replace('+00:00', 'Z')}_{config_hash[:10]}"
    return {
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp_utc,
        "git_commit": commit,
        "git_branch": branch,
        "config_hash": config_hash,
        "runtime_config": runtime_config,
    }


def _attach_metadata(frame, metadata):
    tagged = frame.copy()
    tagged["run_id"] = metadata["run_id"]
    tagged["run_timestamp_utc"] = metadata["run_timestamp_utc"]
    tagged["git_commit"] = metadata["git_commit"]
    tagged["git_branch"] = metadata["git_branch"]
    tagged["config_hash"] = metadata["config_hash"]
    return tagged


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "Input"
    output_dir = project_root / "Output"
    output_dir.mkdir(exist_ok=True)

    input_frames = load_inputs(input_dir)
    prepared = prepare_model_data(input_frames)

    params = fit_structural_model(prepared.x, prepared.y_mid, prepared.y_term, prepared.sigma, prepared.features, prepared.duration)
    baseline_params = fit_baseline_model(prepared.x, prepared.y_mid, prepared.y_term, prepared.sigma)

    mid_pred, term_pred = predict_mid_term(params, prepared.x, prepared.sigma, prepared.features, prepared.duration)
    base_mid_pred, base_term_pred = baseline_predict(baseline_params, prepared.x, prepared.sigma)
    mid_rmse, mid_mae, mid_directional_error = compute_metrics(prepared.y_mid, mid_pred)
    term_rmse, term_mae, term_directional_error = compute_metrics(prepared.y_term, term_pred)
    overall_rmse, overall_mae, overall_directional_error = compute_metrics(
        np.concatenate([prepared.y_mid, prepared.y_term]),
        np.concatenate([mid_pred, term_pred]),
    )

    bootstrap_iterations = int(os.environ.get("IMPACT_BOOTSTRAP_ITERATIONS", "20"))
    runtime_config = {
        "bootstrap_iterations": bootstrap_iterations,
        "execution": {
            "lambda_risk": 1e-6,
            "max_participation": 0.25,
            "n_slices": 13,
            "adaptive_participation": True,
            "kernel_mode": "hybrid",
            "power_alpha": 1.25,
            "kernel_mix": 0.55,
            "spread_penalty_scale": 0.02,
        },
        "portfolio": {
            "lambda_port": 1e-6,
            "max_assets": 5,
        },
        "walkforward": {
            "default_lookback_days": 60,
            "short_lookback_days": 20,
            "recalibration_threshold": 1.15,
            "min_train_days_fraction": 0.35,
            "min_train_observations": 300,
            "min_test_observations": 20,
        },
    }
    runtime_metadata = _build_runtime_metadata(project_root, runtime_config)

    walkforward_result = run_walkforward_analysis(prepared)

    ci_lower, ci_upper, bootstrap_params = run_bootstrap(prepared, bootstrap_iterations, seed=42)
    regime_performance = compute_regime_performance(prepared, mid_pred, term_pred)

    diagnostics_df = _attach_metadata(build_residual_diagnostics(prepared, mid_pred, term_pred), runtime_metadata)
    diagnostics_df.to_csv(output_dir / "residual_diagnostics.csv", index=False)
    _attach_metadata(walkforward_result.walkforward_df, runtime_metadata).to_csv(output_dir / "walkforward_metrics.csv", index=False)
    _attach_metadata(walkforward_result.comparison_df, runtime_metadata).to_csv(output_dir / "oos_model_comparison.csv", index=False)
    _attach_metadata(walkforward_result.policy_df, runtime_metadata).to_csv(output_dir / "execution_policy_comparison.csv", index=False)
    _attach_metadata(walkforward_result.online_df, runtime_metadata).to_csv(output_dir / "online_recalibration.csv", index=False)

    portfolio_df = _attach_metadata(build_latest_portfolio_schedule(prepared, params), runtime_metadata)
    portfolio_df.to_csv(output_dir / "portfolio_schedule.csv", index=False)

    summary = {
        "in_sample": {
            "eta_buy": float(params[0]),
            "eta_sell": float(params[1]),
            "beta": float(params[2]),
            "phi_buy": float(params[3]),
            "phi_sell": float(params[4]),
            "delta": float(params[5]),
            "kappa": float(params[6]),
            "rho": float(params[7]),
            "rmse_mid": mid_rmse,
            "mae_mid": mid_mae,
            "directional_mid": mid_directional_error,
            "rmse_term": term_rmse,
            "mae_term": term_mae,
            "directional_term": term_directional_error,
            "rmse_joint": overall_rmse,
            "mae_joint": overall_mae,
            "directional_joint": overall_directional_error,
        },
        "baseline_in_sample": {
            "eta": float(baseline_params[0]),
            "beta": float(baseline_params[1]),
            "phi": float(baseline_params[2]),
            "delta": float(baseline_params[3]),
            "rmse_joint": float(
                compute_metrics(
                    np.concatenate([prepared.y_mid, prepared.y_term]),
                    np.concatenate([base_mid_pred, base_term_pred]),
                )[0]
            ),
        },
        "bootstrap_ci_95": {
            "eta_buy": [float(ci_lower[0]), float(ci_upper[0])],
            "eta_sell": [float(ci_lower[1]), float(ci_upper[1])],
            "beta": [float(ci_lower[2]), float(ci_upper[2])],
            "phi_buy": [float(ci_lower[3]), float(ci_upper[3])],
            "phi_sell": [float(ci_lower[4]), float(ci_upper[4])],
            "delta": [float(ci_lower[5]), float(ci_upper[5])],
            "kappa": [float(ci_lower[6]), float(ci_upper[6])],
            "rho": [float(ci_lower[7]), float(ci_upper[7])],
            "bootstrap_successful_runs": int(len(bootstrap_params)),
        },
        "walkforward_oos": walkforward_result.oos_summary,
        "oos_model_comparison": walkforward_result.comparison_summary,
        "policy_benchmarks": walkforward_result.policy_summary,
        "regime_performance": regime_performance,
        "online_recalibration": {
            "trigger_count": int(np.sum(walkforward_result.online_df["recalibration_trigger"])) if not walkforward_result.online_df.empty else 0,
            "evaluated_days": int(len(walkforward_result.online_df)),
        },
        "preprocessing": {
            "x_bounds": [float(prepared.x_bounds[0]), float(prepared.x_bounds[1])],
            "y_mid_bounds": [float(prepared.y_mid_bounds[0]), float(prepared.y_mid_bounds[1])],
            "y_term_bounds": [float(prepared.y_term_bounds[0]), float(prepared.y_term_bounds[1])],
            "filtered_observations": int(prepared.x_filtered.size),
            "used_observations": int(prepared.x.size),
        },
        "runtime": runtime_metadata,
    }

    with open(output_dir / "model_diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        "Structural params eta_buy %.5f eta_sell %.5f beta %.5f phi_buy %.5f phi_sell %.5f delta %.5f kappa %.5f rho %.5f"
        % (params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
    )
    print(
        "In-sample joint RMSE %.6f MAE %.6f Directional Error %.6f"
        % (overall_rmse, overall_mae, overall_directional_error)
    )
    print(
        "Walk-forward joint RMSE %.6f MAE %.6f Directional Error %.6f"
        % (
            walkforward_result.oos_summary["rmse_joint"],
            walkforward_result.oos_summary["mae_joint"],
            walkforward_result.oos_summary["directional_joint"],
        )
    )
    if not walkforward_result.policy_df.empty:
        twap_mask = walkforward_result.policy_df["policy"] == "twap"
        vwap_mask = walkforward_result.policy_df["policy"] == "vwap"
        avg_saving_twap = float(np.nanmean(walkforward_result.policy_df.loc[twap_mask, "saving_vs_optimized"])) if np.any(twap_mask) else float("nan")
        avg_saving_vwap = float(np.nanmean(walkforward_result.policy_df.loc[vwap_mask, "saving_vs_optimized"])) if np.any(vwap_mask) else float("nan")
        print(
            "Average benchmark cost premium vs optimized: TWAP %.6f and VWAP %.6f"
            % (
                avg_saving_twap,
                avg_saving_vwap,
            )
        )
    print(f"Diagnostics saved to: {output_dir}")
