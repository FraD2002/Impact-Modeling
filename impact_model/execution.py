import numpy as np
from scipy.optimize import minimize

from impact_model.calibration import split_params


def make_volume_profile(n_slices):
    grid = np.linspace(0.0, 1.0, n_slices)
    profile = 1.0 + 0.6 * np.cos(2.0 * np.pi * (grid - 0.5))
    profile = np.maximum(profile, 0.05)
    profile /= np.sum(profile)
    return profile


def build_schedule_features(profile, liq_z_day, vol_z_day):
    seasonality = (profile - np.mean(profile)) / (np.std(profile) + 1e-12)
    return np.column_stack(
        [
            np.full(profile.size, liq_z_day, dtype=float),
            np.full(profile.size, vol_z_day, dtype=float),
            seasonality,
        ]
    )


def effective_participation_cap(base_cap, liq_z_day, vol_z_day, adaptive=False):
    if not adaptive:
        return float(base_cap)
    multiplier = float(np.exp(np.clip(0.12 * liq_z_day - 0.18 * vol_z_day, -0.7, 0.7)))
    return float(np.clip(base_cap * multiplier, 0.05, 0.35))


def build_transient_kernel(n_slices, rho, kernel_mode="exponential", power_alpha=1.25, kernel_mix=0.5):
    lag = np.abs(np.subtract.outer(np.arange(n_slices), np.arange(n_slices)))
    exp_kernel = np.exp(-rho * lag)
    if kernel_mode == "exponential":
        return exp_kernel
    pow_kernel = np.power(1.0 + lag, -max(power_alpha, 1e-6))
    if kernel_mode == "power_law":
        return pow_kernel
    mix = float(np.clip(kernel_mix, 0.0, 1.0))
    return mix * exp_kernel + (1.0 - mix) * pow_kernel


def feasible_profile_schedule(q_abs, upper, profile):
    u = q_abs * profile
    u = np.minimum(u, upper)
    if np.sum(u) <= 1e-12:
        u = np.minimum(np.full_like(upper, q_abs / upper.size), upper)
    if np.sum(u) <= 1e-12:
        return u
    u *= q_abs / np.sum(u)
    u = np.minimum(u, upper)
    deficit = q_abs - np.sum(u)
    for _ in range(30):
        if deficit <= 1e-10:
            break
        capacity = upper - u
        cap_sum = np.sum(capacity)
        if cap_sum <= 1e-12:
            break
        add = np.minimum(capacity, deficit * capacity / cap_sum)
        u += add
        deficit = q_abs - np.sum(u)
    return u


def schedule_cost_components(
    u,
    Q,
    V_day,
    sigma_day,
    params,
    liq_z_day,
    vol_z_day,
    lambda_risk,
    kernel_mode="exponential",
    power_alpha=1.25,
    kernel_mix=0.5,
    spread_penalty_scale=0.0,
):
    if np.abs(Q) <= 1e-12:
        return {"temp": 0.0, "transient": 0.0, "risk": 0.0, "perm": 0.0, "spread": 0.0, "total": 0.0}

    profile = make_volume_profile(u.size)
    avail = np.maximum(V_day * profile, 1e-12)
    rate = np.maximum(u / avail, 1e-12)
    eta_buy, eta_sell, beta, phi_buy, phi_sell, delta, kappa, rho, theta_temp, theta_perm = split_params(params)
    side = 1.0 if Q >= 0 else -1.0
    eta_side = eta_buy if side > 0 else eta_sell
    phi_side = phi_buy if side > 0 else phi_sell
    features = build_schedule_features(profile, liq_z_day, vol_z_day)
    temp_scale = np.exp(np.clip(features @ theta_temp, -10.0, 10.0))
    perm_scale = np.exp(np.clip(np.array([liq_z_day, vol_z_day, 0.0]) @ theta_perm, -10.0, 10.0))
    temp = sigma_day * eta_side * np.sum(temp_scale * np.power(rate, beta) * u)
    kernel = build_transient_kernel(u.size, rho, kernel_mode=kernel_mode, power_alpha=power_alpha, kernel_mix=kernel_mix)
    transient = sigma_day * kappa * float(np.power(rate, beta) @ kernel @ np.power(rate, beta)) * np.mean(avail)
    q_abs = np.abs(Q)
    q_path = q_abs - np.cumsum(u)
    risk = lambda_risk * (sigma_day**2) * float(np.sum(np.square(q_path)) / u.size)
    perm = sigma_day * phi_side * np.power(q_abs / max(V_day, 1e-12), delta) * q_abs * perm_scale
    spread_proxy = sigma_day / np.sqrt(max(V_day, 1e-12))
    spread = spread_penalty_scale * spread_proxy * float(np.sum(u))
    total = temp + transient + risk + perm + spread
    return {"temp": float(temp), "transient": float(transient), "risk": float(risk), "perm": float(perm), "spread": float(spread), "total": float(total)}


def optimize_schedule_single_asset(
    Q,
    V_day,
    sigma_day,
    params,
    liq_z_day,
    vol_z_day,
    lambda_risk=1e-6,
    max_participation=0.25,
    n_slices=13,
    adaptive_participation=False,
    kernel_mode="exponential",
    power_alpha=1.25,
    kernel_mix=0.5,
    spread_penalty_scale=0.0,
):
    if not np.isfinite(Q) or not np.isfinite(V_day) or not np.isfinite(sigma_day):
        return np.zeros(n_slices, dtype=float), {"temp": 0.0, "transient": 0.0, "risk": 0.0, "perm": 0.0, "spread": 0.0, "total": 0.0}
    if V_day <= 0 or sigma_day <= 0:
        return np.zeros(n_slices, dtype=float), {"temp": 0.0, "transient": 0.0, "risk": 0.0, "perm": 0.0, "spread": 0.0, "total": 0.0}

    side = 1.0 if Q >= 0 else -1.0
    q_abs = np.abs(Q)
    profile = make_volume_profile(n_slices)
    avail = np.maximum(V_day * profile, 1e-12)
    cap = effective_participation_cap(max_participation, liq_z_day, vol_z_day, adaptive=adaptive_participation)
    upper = cap * avail
    max_feasible = float(np.sum(upper))
    if max_feasible <= 1e-12:
        return np.zeros(n_slices, dtype=float), {"temp": 0.0, "transient": 0.0, "risk": 0.0, "perm": 0.0, "spread": 0.0, "total": 0.0}
    q_abs = min(q_abs, 0.98 * max_feasible)
    u0 = feasible_profile_schedule(q_abs, upper, profile)

    def objective(u):
        return schedule_cost_components(
            u,
            side * q_abs,
            V_day,
            sigma_day,
            params,
            liq_z_day,
            vol_z_day,
            lambda_risk,
            kernel_mode=kernel_mode,
            power_alpha=power_alpha,
            kernel_mix=kernel_mix,
            spread_penalty_scale=spread_penalty_scale,
        )["total"]

    constraints = [{"type": "eq", "fun": lambda u: np.sum(u) - q_abs}]
    bounds = [(0.0, float(upper_i)) for upper_i in upper]
    result = minimize(
        objective,
        x0=u0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    if result.success and np.all(np.isfinite(result.x)):
        u_opt = result.x
    else:
        u_opt = u0
    costs = schedule_cost_components(
        u_opt,
        side * q_abs,
        V_day,
        sigma_day,
        params,
        liq_z_day,
        vol_z_day,
        lambda_risk,
        kernel_mode=kernel_mode,
        power_alpha=power_alpha,
        kernel_mix=kernel_mix,
        spread_penalty_scale=spread_penalty_scale,
    )
    return side * u_opt, costs


def twap_schedule(Q, n_slices):
    if np.abs(Q) <= 1e-12:
        return np.zeros(n_slices, dtype=float)
    return np.full(n_slices, Q / n_slices, dtype=float)


def vwap_schedule(Q, n_slices):
    profile = make_volume_profile(n_slices)
    return Q * profile


def pov_schedule(Q, V_day, max_participation, n_slices):
    side = 1.0 if Q >= 0 else -1.0
    q_abs = np.abs(Q)
    profile = make_volume_profile(n_slices)
    avail = np.maximum(V_day * profile, 1e-12)
    upper = max_participation * avail
    q_abs = min(q_abs, 0.98 * float(np.sum(upper)))
    schedule = feasible_profile_schedule(q_abs, upper, profile)
    return side * schedule


def front_loaded_schedule(Q, n_slices, decay=0.25):
    if np.abs(Q) <= 1e-12:
        return np.zeros(n_slices, dtype=float)
    idx = np.arange(n_slices, dtype=float)
    weights = np.exp(-decay * idx)
    weights /= np.sum(weights)
    return Q * weights


def portfolio_objective(
    flat_u,
    q_abs_vec,
    side_vec,
    V_vec,
    sigma_vec,
    params_vec,
    liq_z_vec,
    vol_z_vec,
    cov_matrix,
    cross_matrix,
    lambda_risk,
    lambda_port,
    max_participation,
    n_slices,
    kernel_mode="exponential",
    power_alpha=1.25,
    kernel_mix=0.5,
    spread_penalty_scale=0.0,
):
    m = q_abs_vec.size
    u = flat_u.reshape(m, n_slices)
    total = 0.0
    profile = make_volume_profile(n_slices)
    avail = np.maximum(np.outer(V_vec, profile), 1e-12)
    rates_signed = np.where(avail > 0, (u / avail) * side_vec[:, None], 0.0)
    for i in range(m):
        components = schedule_cost_components(
            u[i],
            side_vec[i] * q_abs_vec[i],
            V_vec[i],
            sigma_vec[i],
            params_vec[i],
            liq_z_vec[i],
            vol_z_vec[i],
            lambda_risk,
            kernel_mode=kernel_mode,
            power_alpha=power_alpha,
            kernel_mix=kernel_mix,
            spread_penalty_scale=spread_penalty_scale,
        )
        total += components["total"]
    for t in range(n_slices):
        r_t = rates_signed[:, t]
        total += float(r_t @ cross_matrix @ r_t) * float(np.mean(V_vec))
    q_path = q_abs_vec[:, None] - np.cumsum(u, axis=1)
    q_signed = q_path * side_vec[:, None]
    for t in range(n_slices):
        total += lambda_port * float(q_signed[:, t] @ cov_matrix @ q_signed[:, t]) / n_slices
    return total


def optimize_portfolio_schedule(
    Q_vec,
    V_vec,
    sigma_vec,
    params_vec,
    liq_z_vec,
    vol_z_vec,
    cov_matrix,
    cross_matrix,
    lambda_risk=1e-6,
    lambda_port=1e-6,
    max_participation=0.25,
    n_slices=13,
    adaptive_participation=False,
    kernel_mode="exponential",
    power_alpha=1.25,
    kernel_mix=0.5,
    spread_penalty_scale=0.0,
):
    side_vec = np.where(Q_vec >= 0, 1.0, -1.0)
    q_abs_vec = np.abs(Q_vec)
    profile = make_volume_profile(n_slices)
    avail = np.maximum(np.outer(V_vec, profile), 1e-12)
    caps = np.array(
        [
            effective_participation_cap(max_participation, liq_z_vec[i], vol_z_vec[i], adaptive=adaptive_participation)
            for i in range(q_abs_vec.size)
        ],
        dtype=float,
    )
    upper = caps[:, None] * avail
    max_feasible = np.sum(upper, axis=1)
    q_abs_vec = np.minimum(q_abs_vec, 0.98 * max_feasible)
    u0 = np.zeros((q_abs_vec.size, n_slices), dtype=float)
    for i in range(q_abs_vec.size):
        u0[i] = feasible_profile_schedule(q_abs_vec[i], upper[i], profile)

    constraints = []
    for i in range(q_abs_vec.size):
        constraints.append({"type": "eq", "fun": lambda flat_u, i=i: np.sum(flat_u.reshape(q_abs_vec.size, n_slices)[i]) - q_abs_vec[i]})
    bounds = [(0.0, float(upper[i, t])) for i in range(q_abs_vec.size) for t in range(n_slices)]

    result = minimize(
        portfolio_objective,
        x0=u0.ravel(),
        args=(
            q_abs_vec,
            side_vec,
            V_vec,
            sigma_vec,
            params_vec,
            liq_z_vec,
            vol_z_vec,
            cov_matrix,
            cross_matrix,
            lambda_risk,
            lambda_port,
            max_participation,
            n_slices,
            kernel_mode,
            power_alpha,
            kernel_mix,
            spread_penalty_scale,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )
    if result.success and np.all(np.isfinite(result.x)):
        u_opt = result.x.reshape(q_abs_vec.size, n_slices)
    else:
        u_opt = u0
    schedule = u_opt * side_vec[:, None]
    return schedule
