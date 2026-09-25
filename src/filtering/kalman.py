import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal


def kalman_filter(F, H, y, x0_mean, x0_cov, Q, R):
    """
    Exact Kalman filter for linear-Gaussian SSMs:

        x_{k+1} = F x_k + v_k,  v_k ~ N(0, Q)
        y_k     = H x_k + w_k,  w_k ~ N(0, R)

    Args:
        F:        transition matrix, shape (D, D)
        H:        observation matrix, shape (O, D)
        y:        observations, shape (T, O)
        x0_mean:  prior mean, shape (D,)
        x0_cov:   prior covariance, shape (D, D)
        Q:        process noise covariance, shape (D, D)
        R:        observation noise covariance, shape (O, O)

    Returns:
        means:                   filtered means, shape (T, D)
        covs:                    filtered covariances, shape (T, D, D)
        log_marginal_likelihood: scalar
    """
    D = x0_mean.shape[0]
    I = jnp.eye(D)

    def update(x_pred, P_pred, y_k):
        S = H @ P_pred @ H.T + R                          # (O, O) innovation covariance
        K = jnp.linalg.solve(S.T, (H @ P_pred).T).T      # (D, O) Kalman gain
        innovation = y_k - H @ x_pred                     # (O,)
        x_new = x_pred + K @ innovation                   # (D,)
        ImKH = I - K @ H
        P_new = ImKH @ P_pred @ ImKH.T + K @ R @ K.T     # (D, D) Joseph form
        log_lik = multivariate_normal.logpdf(y_k, H @ x_pred, S)
        return x_new, P_new, log_lik

    def step(carry, y_k):
        x, P = carry
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        x_new, P_new, log_lik = update(x_pred, P_pred, y_k)
        return (x_new, P_new), (x_new, P_new, log_lik)

    x0_filt, P0_filt, log_lik_0 = update(x0_mean, x0_cov, y[0])

    _, (means, covs, log_liks) = jax.lax.scan(step, (x0_filt, P0_filt), y[1:])

    means = jnp.concatenate([x0_filt[None], means], axis=0)
    covs = jnp.concatenate([P0_filt[None], covs], axis=0)
    log_marginal_likelihood = log_lik_0 + jnp.sum(log_liks)

    return means, covs, log_marginal_likelihood


def unscented_kalman_filter(
    transition,
    observation,
    t,
    y,
    x0_mean,
    x0_cov,
    Q,
    R,
    *,
    n_substeps=1,
    alpha=1.0,
    beta=2.0,
    kappa=0.0,
):
    """
    Unscented Kalman filter (UKF) for nonlinear-Gaussian SSMs:

        x_{k+1} = transition(x_k, t_k, dt) + v_k,  v_k ~ N(0, Q)
        y_k     = observation(x_k) + w_k,            w_k ~ N(0, R)

    Propagates 2D+1 sigma points through the nonlinear functions to
    approximate the predicted mean and covariance at each step.

    Args:
        transition:  f(x, t, dt) -> x_next, supports batched x of shape (..., D)
        observation: h(x) -> y, supports batched x of shape (..., D)
        t:           observation times, shape (T,)
        y:           observations, shape (T, O)
        x0_mean:     prior mean, shape (D,)
        x0_cov:      prior covariance, shape (D, D)
        Q:           process noise covariance, shape (D, D)
        R:           observation noise covariance, shape (O, O)
        n_substeps:  UKF prediction substeps per observation interval (default 1)
        alpha:       sigma point spread (default 1.0; use ~1e-3 for Gaussian priors)
        beta:        distribution prior parameter (default 2.0, optimal for Gaussian)
        kappa:       secondary scaling (default 0.0)

    Returns:
        means:                   filtered means, shape (T, D)
        covs:                    filtered covariances, shape (T, D, D)
        log_marginal_likelihood: scalar
    """
    D = x0_mean.shape[0]
    lambda_ = alpha**2 * (D + kappa) - D
    c = D + lambda_

    # Sigma point weights
    W_m = jnp.concatenate([jnp.array([lambda_ / c]), jnp.full(2 * D, 0.5 / c)])
    W_c = W_m.at[0].set(lambda_ / c + (1.0 - alpha**2 + beta))

    def sigma_points(x, P):
        """Generate 2D+1 sigma points from mean x and covariance P."""
        L = jnp.linalg.cholesky(c * P)       # (D, D) lower triangular
        return jnp.concatenate([x[None, :], x + L.T, x - L.T], axis=0)  # (2D+1, D)

    def predict(x, P, t_start, dt_obs):
        """Multi-substep UKF prediction."""
        dt_sub = dt_obs / n_substeps
        t_sub = t_start + jnp.arange(n_substeps) * dt_sub

        def substep(carry, t_k):
            x, P = carry
            sp = sigma_points(x, P)                           # (2D+1, D)
            sp_prop = transition(sp, t_k, dt_sub)             # (2D+1, D)
            x_pred = W_m @ sp_prop                            # (D,)
            diff = sp_prop - x_pred                           # (2D+1, D)
            P_pred = jnp.einsum("n,nd,ne->de", W_c, diff, diff) + Q  # (D, D)
            return (x_pred, P_pred), None

        (x_pred, P_pred), _ = jax.lax.scan(substep, (x, P), t_sub)
        return x_pred, P_pred

    def update(x_pred, P_pred, y_k):
        sp = sigma_points(x_pred, P_pred)                     # (2D+1, D)
        y_sp = observation(sp)                                 # (2D+1, O)
        y_pred = W_m @ y_sp                                   # (O,)

        diff_x = sp - x_pred                                  # (2D+1, D)
        diff_y = y_sp - y_pred                                # (2D+1, O)

        S = jnp.einsum("n,no,np->op", W_c, diff_y, diff_y) + R  # (O, O)
        P_xy = jnp.einsum("n,nd,no->do", W_c, diff_x, diff_y)   # (D, O)

        K = jnp.linalg.solve(S.T, P_xy.T).T                  # (D, O)
        innovation = y_k - y_pred                             # (O,)
        x_new = x_pred + K @ innovation                       # (D,)
        P_new = P_pred - K @ S @ K.T                          # (D, D)
        log_lik = multivariate_normal.logpdf(y_k, y_pred, S)
        return x_new, P_new, log_lik

    def step(carry, inputs):
        x, P = carry
        t_k, t_next, y_k = inputs
        x_pred, P_pred = predict(x, P, t_k, t_next - t_k)
        x_new, P_new, log_lik = update(x_pred, P_pred, y_k)
        return (x_new, P_new), (x_new, P_new, log_lik)

    x0_filt, P0_filt, log_lik_0 = update(x0_mean, x0_cov, y[0])

    _, (means, covs, log_liks) = jax.lax.scan(
        step, (x0_filt, P0_filt), (t[:-1], t[1:], y[1:])
    )

    means = jnp.concatenate([x0_filt[None], means], axis=0)
    covs = jnp.concatenate([P0_filt[None], covs], axis=0)
    log_marginal_likelihood = log_lik_0 + jnp.sum(log_liks)

    return means, covs, log_marginal_likelihood
