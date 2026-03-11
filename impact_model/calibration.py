import numpy as np
from scipy.optimize import least_squares


def zscore(values):
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values, dtype=float), float(mean), float(1.0)
    return (values - mean) / std, float(mean), float(std)


def get_clip_bounds(values):
    lower = float(np.quantile(values, 0.01))
    upper = float(np.quantile(values, 0.99))
    return lower, upper


def apply_clips(values, bounds):
    return np.clip(values, bounds[0], bounds[1])


def build_state_features(V_matrix, sigma_matrix, day_idx_matrix):
    liq = np.log(np.maximum(V_matrix, 1e-12))
    liq_z, liq_mean, liq_std = zscore(liq)
    vol_z, vol_mean, vol_std = zscore(sigma_matrix)
    seasonality = np.sin(2.0 * np.pi * ((day_idx_matrix % 5) / 5.0))
    features = np.stack([liq_z, vol_z, seasonality], axis=-1)
    feature_context = {
        "liq_mean": liq_mean,
        "liq_std": liq_std,
        "vol_mean": vol_mean,
        "vol_std": vol_std,
    }
    return features, feature_context


def split_params(params):
    eta_buy, eta_sell, beta, phi_buy, phi_sell, delta, kappa, rho = params[:8]
    theta_temp = params[8:11]
    theta_perm = params[11:14]
    return eta_buy, eta_sell, beta, phi_buy, phi_sell, delta, kappa, rho, theta_temp, theta_perm


def predict_mid_term(params, x, sigma, features, duration):
    eta_buy, eta_sell, beta, phi_buy, phi_sell, delta, kappa, rho, theta_temp, theta_perm = split_params(params)
    side = np.sign(x)
    abs_x = np.abs(x)
    eta = np.where(x >= 0, eta_buy, eta_sell)
    phi = np.where(x >= 0, phi_buy, phi_sell)
    temp_scale = np.exp(np.clip(features @ theta_temp, -10.0, 10.0))
    perm_scale = np.exp(np.clip(features @ theta_perm, -10.0, 10.0))
    duration_scale = np.exp(-rho * duration)
    temp = sigma * eta * side * np.power(abs_x, beta) * temp_scale
    transient = sigma * kappa * side * np.power(abs_x, beta) * temp_scale * duration_scale
    perm = sigma * phi * side * np.power(abs_x, delta) * perm_scale
    mid_pred = temp + transient + 0.5 * perm
    term_pred = perm
    return mid_pred, term_pred


def fit_structural_model(x, y_mid, y_term, sigma, features, duration, prior_params=None, smoothing_weight=0.0):
    weights = 1.0 / np.maximum(sigma, 1e-12)

    def residuals(params):
        mid_pred, term_pred = predict_mid_term(params, x, sigma, features, duration)
        mid_res = np.sqrt(weights) * (y_mid - mid_pred)
        term_res = np.sqrt(weights) * (y_term - term_pred)
        if prior_params is not None and smoothing_weight > 0:
            prior_res = np.sqrt(smoothing_weight) * (params - prior_params)
            return np.concatenate([mid_res, term_res, prior_res])
        return np.concatenate([mid_res, term_res])

    x0 = np.array([0.3, 0.3, 0.35, 0.1, 0.1, 0.35, 0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    if prior_params is not None:
        x0 = np.array(prior_params, dtype=float)
    lower_bounds = np.array([1e-12, 1e-12, 1e-8, 1e-12, 1e-12, 1e-8, 0.0, 1e-8, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], dtype=float)
    upper_bounds = np.array([np.inf, np.inf, 0.999999, np.inf, np.inf, 0.999999, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=8000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def baseline_predict(params, x, sigma):
    eta, beta, phi, delta = params
    side = np.sign(x)
    abs_x = np.abs(x)
    temp = sigma * eta * side * np.power(abs_x, beta)
    perm = sigma * phi * side * np.power(abs_x, delta)
    mid_pred = temp + 0.5 * perm
    term_pred = perm
    return mid_pred, term_pred


def fit_baseline_model(x, y_mid, y_term, sigma):
    weights = 1.0 / np.maximum(sigma, 1e-12)

    def residuals(params):
        mid_pred, term_pred = baseline_predict(params, x, sigma)
        mid_res = np.sqrt(weights) * (y_mid - mid_pred)
        term_res = np.sqrt(weights) * (y_term - term_pred)
        return np.concatenate([mid_res, term_res])

    x0 = np.array([0.3, 0.35, 0.1, 0.35], dtype=float)
    lower_bounds = np.array([1e-12, 1e-8, 1e-12, 1e-8], dtype=float)
    upper_bounds = np.array([np.inf, 0.999999, np.inf, 0.999999], dtype=float)
    result = least_squares(
        residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    valid = sign_true != 0
    if np.any(valid):
        directional_error = float(np.mean(sign_true[valid] != sign_pred[valid]))
    else:
        directional_error = float("nan")
    return rmse, mae, directional_error


def classify_regime(train_sigma_values, test_sigma_value):
    q1, q2 = np.quantile(train_sigma_values, [0.33, 0.66])
    if test_sigma_value <= q1:
        return "low_vol"
    if test_sigma_value <= q2:
        return "mid_vol"
    return "high_vol"


def clip_param_drift(params, prior):
    drift = np.array([0.25, 0.25, 0.15, 0.25, 0.25, 0.15, 0.25, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=float)
    lower = prior - drift
    upper = prior + drift
    return np.minimum(np.maximum(params, lower), upper)
