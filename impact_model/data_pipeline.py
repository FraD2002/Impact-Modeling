from dataclasses import dataclass

import numpy as np
import pandas as pd

from impact_model.calibration import apply_clips, build_state_features, get_clip_bounds
from project_io import drop_columns_if_present, read_required_csvs


@dataclass
class InputFrames:
    totalDailyValue: pd.DataFrame
    imbalance: pd.DataFrame
    vwap330: pd.DataFrame
    arrivalPrice: pd.DataFrame
    terminalPrice: pd.DataFrame
    dailyVol: pd.DataFrame


@dataclass
class PreparedModelData:
    x: np.ndarray
    y_mid: np.ndarray
    y_term: np.ndarray
    sigma: np.ndarray
    day_idx: np.ndarray
    asset_idx: np.ndarray
    features: np.ndarray
    duration: np.ndarray
    x_bounds: tuple[float, float]
    y_mid_bounds: tuple[float, float]
    y_term_bounds: tuple[float, float]
    x_filtered: np.ndarray
    y_mid_filtered: np.ndarray
    y_term_filtered: np.ndarray
    sigma_filtered: np.ndarray
    day_idx_filtered: np.ndarray
    asset_idx_filtered: np.ndarray
    features_filtered: np.ndarray
    x_matrix: np.ndarray
    y_mid_matrix: np.ndarray
    y_term_matrix: np.ndarray
    sigma_matrix: np.ndarray
    V_matrix: np.ndarray
    day_idx_matrix: np.ndarray
    asset_idx_matrix: np.ndarray
    y_mid_df: pd.DataFrame
    feature_context: dict


def load_inputs(input_dir):
    required_files = [
        "totalDailyValueDf.csv",
        "imbalanceDf.csv",
        "vwap330Df.csv",
        "arrivalPriceDf.csv",
        "terminalPriceDf.csv",
        "dailyVolDf.csv",
    ]
    frames = read_required_csvs(input_dir, required_files)
    return InputFrames(
        totalDailyValue=drop_columns_if_present(frames["totalDailyValueDf"], ["Stock"]),
        imbalance=drop_columns_if_present(frames["imbalanceDf"], ["Stock"]),
        vwap330=drop_columns_if_present(frames["vwap330Df"], ["Stock"]),
        arrivalPrice=drop_columns_if_present(frames["arrivalPriceDf"], ["Stock"]),
        terminalPrice=drop_columns_if_present(frames["terminalPriceDf"], ["Stock"]),
        dailyVol=drop_columns_if_present(frames["dailyVolDf"], ["Stock"]),
    )


def rolling_average_daily_value(totalDailyValue, window=10):
    return totalDailyValue.T.rolling(window=window, min_periods=window).mean().T.fillna(0.0)


def prepare_model_data(input_frames, warmup_days=10, min_observations=200):
    totalDailyValue = input_frames.totalDailyValue.copy()
    imbalance = input_frames.imbalance.copy()
    vwap330 = input_frames.vwap330.copy()
    arrivalPrice = input_frames.arrivalPrice.copy()
    terminalPrice = input_frames.terminalPrice.copy()
    dailyVol = input_frames.dailyVol.copy()

    avgDailyValue = rolling_average_daily_value(totalDailyValue, window=warmup_days)

    totalDailyValue = totalDailyValue.iloc[:, warmup_days:]
    imbalance = imbalance.iloc[:, warmup_days:]
    vwap330 = vwap330.iloc[:, warmup_days:]
    arrivalPrice = arrivalPrice.iloc[:, warmup_days:]
    terminalPrice = terminalPrice.iloc[:, warmup_days:]
    dailyVol = dailyVol.iloc[:, warmup_days:]
    avgDailyValue = avgDailyValue.iloc[:, warmup_days:]

    X = imbalance
    V = avgDailyValue
    y_mid_df = vwap330 - arrivalPrice
    y_term_df = terminalPrice - arrivalPrice

    x_matrix = (X / (6 * V / 6.5)).to_numpy(dtype=float)
    y_mid_matrix = y_mid_df.to_numpy(dtype=float)
    y_term_matrix = y_term_df.to_numpy(dtype=float)
    sigma_matrix = dailyVol.to_numpy(dtype=float)
    V_matrix = V.to_numpy(dtype=float)
    day_idx_matrix = np.broadcast_to(np.arange(x_matrix.shape[1]), x_matrix.shape)
    asset_idx_matrix = np.broadcast_to(np.arange(x_matrix.shape[0])[:, None], x_matrix.shape)
    features_matrix, feature_context = build_state_features(V_matrix, sigma_matrix, day_idx_matrix)

    mask = (
        np.isfinite(x_matrix)
        & np.isfinite(y_mid_matrix)
        & np.isfinite(y_term_matrix)
        & np.isfinite(sigma_matrix)
        & np.isfinite(V_matrix)
        & (V_matrix > 0)
        & (sigma_matrix > 0)
    )

    x_filtered = x_matrix[mask]
    y_mid_filtered = y_mid_matrix[mask]
    y_term_filtered = y_term_matrix[mask]
    sigma_filtered = sigma_matrix[mask]
    day_idx_filtered = day_idx_matrix[mask].astype(int)
    asset_idx_filtered = asset_idx_matrix[mask].astype(int)
    features_filtered = features_matrix[mask]

    if x_filtered.size < min_observations:
        raise ValueError("Not enough valid observations after filtering.")

    x_bounds = get_clip_bounds(x_filtered)
    y_mid_bounds = get_clip_bounds(y_mid_filtered)
    y_term_bounds = get_clip_bounds(y_term_filtered)
    x = apply_clips(x_filtered, x_bounds)
    y_mid = apply_clips(y_mid_filtered, y_mid_bounds)
    y_term = apply_clips(y_term_filtered, y_term_bounds)
    sigma = sigma_filtered
    day_idx = day_idx_filtered
    asset_idx = asset_idx_filtered
    features = features_filtered
    duration = np.clip(np.abs(x) / (np.quantile(np.abs(x), 0.75) + 1e-12), 0.0, 5.0)

    return PreparedModelData(
        x=x,
        y_mid=y_mid,
        y_term=y_term,
        sigma=sigma,
        day_idx=day_idx,
        asset_idx=asset_idx,
        features=features,
        duration=duration,
        x_bounds=x_bounds,
        y_mid_bounds=y_mid_bounds,
        y_term_bounds=y_term_bounds,
        x_filtered=x_filtered,
        y_mid_filtered=y_mid_filtered,
        y_term_filtered=y_term_filtered,
        sigma_filtered=sigma_filtered,
        day_idx_filtered=day_idx_filtered,
        asset_idx_filtered=asset_idx_filtered,
        features_filtered=features_filtered,
        x_matrix=x_matrix,
        y_mid_matrix=y_mid_matrix,
        y_term_matrix=y_term_matrix,
        sigma_matrix=sigma_matrix,
        V_matrix=V_matrix,
        day_idx_matrix=day_idx_matrix,
        asset_idx_matrix=asset_idx_matrix,
        y_mid_df=y_mid_df,
        feature_context=feature_context,
    )
