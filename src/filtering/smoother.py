import jax
import jax.numpy as jnp


def particle_smoother(
    transition, particles, log_weights, t, Q, key, *, n_trajectories=1
):
    """
    Forward-filtering backward-sampling (FFBS) particle smoother.

    Given the output of particle_filter, samples n_trajectories smoothed state
    trajectories from an approximation of p(x_{0:T-1} | y_{0:T-1}).

    The backward kernel uses the single-step Gaussian transition density:

        p(x_{t+1} | x_t) = N(x_{t+1}; transition(x_t, t_k, dt), Q)

    This is exact when n_substeps=1 in the forward filter. When n_substeps > 1,
    pass Q * n_substeps as an approximation of the effective noise over the full
    observation interval.

    Args:
        transition:     f(x, t, dt) -> x_next, supports batched x of shape (..., D)
        particles:      shape (T, N, D) from particle_filter
        log_weights:    shape (T, N) normalised log weights from particle_filter
        t:              observation times, shape (T,)
        Q:              process noise covariance for backward kernel, shape (D, D)
        key:            JAX random key
        n_trajectories: number of smoothed trajectories to draw (default 1)

    Returns:
        trajectories: shape (n_trajectories, T, D)
    """
    T, N, _ = particles.shape
    Q_inv = jnp.linalg.inv(Q)

    # --- Sample terminal states from the filtering distribution at t=T-1 ---
    key, key_T = jax.random.split(key)
    idx_T = jax.random.categorical(
        key_T,
        jnp.broadcast_to(log_weights[-1][None, :], (n_trajectories, N)),
        axis=-1,
    )  # (n_trajectories,)
    x_T = particles[-1][idx_T]  # (n_trajectories, D)

    # --- Backward scan from t=T-2 down to t=0 ---
    rev_idx = jnp.arange(T - 2, -1, -1)  # [T-2, T-3, ..., 0]
    particles_rev = particles[rev_idx]  # (T-1, N, D)
    log_weights_rev = log_weights[rev_idx]  # (T-1, N)
    t_rev = t[rev_idx]  # (T-1,)
    dt_rev = (t[1:] - t[:-1])[rev_idx]  # (T-1,)

    def backward_step(carry, inputs):
        x_next, key = carry  # x_next: (n_trajectories, D)
        particles_t, log_weights_t, t_k, dt = inputs

        key, key_s = jax.random.split(key)

        # Predicted mean from each particle at time t: (N, D)
        x_pred = transition(particles_t, t_k, dt)

        # Mahalanobis distance: diff[j, n, :] = x_next[j] - x_pred[n]
        diff = x_next[:, None, :] - x_pred[None, :, :]  # (n_trajectories, N, D)
        log_trans = -0.5 * jnp.einsum(
            "jnd,de,jne->jn", diff, Q_inv, diff
        )  # (n_traj, N)

        # Backward weights = filtering weight * transition density
        log_bw = log_weights_t[None, :] + log_trans  # (n_trajectories, N)

        # Sample ancestor index for each trajectory
        idx = jax.random.categorical(key_s, log_bw, axis=-1)  # (n_trajectories,)
        x_t = particles_t[idx]  # (n_trajectories, D)

        return (x_t, key), x_t

    _, x_smooth_rev = jax.lax.scan(
        backward_step,
        (x_T, key),
        (particles_rev, log_weights_rev, t_rev, dt_rev),
    )
    # x_smooth_rev: (T-1, n_trajectories, D), time order [T-2, T-3, ..., 0]

    # Reverse to chronological order and append terminal state
    x_all = jnp.concatenate(
        [x_smooth_rev[::-1], x_T[None, :, :]],
        axis=0,
    )  # (T, n_trajectories, D)

    return jnp.swapaxes(x_all, 0, 1)  # (n_trajectories, T, D)
