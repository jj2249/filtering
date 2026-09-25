import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal


def particle_filter(
    transition,
    observation,
    t,
    y,
    n_particles,
    x0_mean,
    x0_cov,
    Q,
    R,
    key,
    *,
    n_substeps=1,
    ess_threshold=0.5,
):
    """
    Bootstrap particle filter for a nonlinear state space model with additive Gaussian noise.

        x_{k+1} = transition(x_k, t_k, dt) + v_k,  v_k ~ N(0, Q)
        y_k     = observation(x_k) + w_k,            w_k ~ N(0, R)

    Args:
        transition:   f(x, t, dt) -> x_next, supports batched x of shape (..., D)
        observation:  h(x) -> y, supports batched x of shape (..., D)
        t:            observation times, shape (T,)
        y:            observations, shape (T, O)
        n_particles:  number of particles
        x0_mean:      prior mean of initial state, shape (D,)
        x0_cov:       prior covariance of initial state, shape (D, D)
        Q:            process noise covariance, shape (D, D)
        R:            observation noise covariance, shape (O, O)
        key:          JAX random key
        n_substeps:   number of transition substeps per observation interval (default 1)
        ess_threshold: resample when ESS falls below this fraction of n_particles (default 0.5)

    Returns:
        particles:             shape (T, N, D)
        log_weights:           shape (T, N)
        ess:                   shape (T,)
        log_marginal_likelihood: scalar estimate of log p(y_{0:T-1})
    """
    D = x0_mean.shape[0]
    ess_threshold_abs = ess_threshold * n_particles

    def log_likelihood(y_t, particles):
        return multivariate_normal.logpdf(y_t, observation(particles), R)

    def normalise(lw):
        return lw - logsumexp(lw)

    def propagate(p, t_start, dt_obs, key):
        dt_sub = dt_obs / n_substeps
        t_sub = t_start + jnp.arange(n_substeps) * dt_sub
        keys = jax.random.split(key, n_substeps)

        def substep(p, inputs):
            t_k, key_k = inputs
            v = jax.random.multivariate_normal(
                key_k, jnp.zeros(D), Q, shape=(n_particles,)
            )
            return transition(p, t_k, dt_sub) + v, None

        p, _ = jax.lax.scan(substep, p, (t_sub, keys))
        return p

    def resample(args):
        p, lw, key = args
        cdf = jnp.cumsum(jnp.exp(lw))
        u = (
            jnp.arange(n_particles) + jax.random.uniform(key, shape=(n_particles,))
        ) / n_particles
        return p[jnp.searchsorted(cdf, u)], jnp.full(n_particles, -jnp.log(n_particles))

    def skip_resample(args):
        p, lw, _ = args
        return p, lw

    def step(carry, inputs):
        p, lw, key = carry
        t_k, t_next, y_next = inputs

        key, key_prop, key_res = jax.random.split(key, 3)

        p = propagate(p, t_k, t_next - t_k, key_prop)
        lw_unnorm = lw + log_likelihood(y_next, p)
        log_Z = logsumexp(lw_unnorm)
        lw = lw_unnorm - log_Z
        ess = ess_from_log_weights(lw)

        p, lw = jax.lax.cond(
            ess < ess_threshold_abs, resample, skip_resample, (p, lw, key_res)
        )

        return (p, lw, key), (p, lw, ess, log_Z)

    key, key_init = jax.random.split(key)
    p0 = jax.random.multivariate_normal(key_init, x0_mean, x0_cov, shape=(n_particles,))
    log_lik_0 = log_likelihood(y[0], p0)
    log_Z_0 = logsumexp(log_lik_0) - jnp.log(n_particles)
    lw0 = normalise(log_lik_0)
    ess0 = ess_from_log_weights(lw0)

    _, (particles, log_weights, ess, log_Zs) = jax.lax.scan(
        step, (p0, lw0, key), (t[:-1], t[1:], y[1:])
    )

    particles = jnp.concatenate([p0[jnp.newaxis], particles], axis=0)
    log_weights = jnp.concatenate([lw0[jnp.newaxis], log_weights], axis=0)
    ess = jnp.concatenate([jnp.array([ess0]), ess], axis=0)
    log_marginal_likelihood = log_Z_0 + jnp.sum(log_Zs)

    return particles, log_weights, ess, log_marginal_likelihood


def auxiliary_particle_filter(
    transition,
    observation,
    t,
    y,
    n_particles,
    x0_mean,
    x0_cov,
    Q,
    R,
    key,
    *,
    n_substeps=1,
    ess_threshold=0.5,
):
    """
    Auxiliary particle filter (APF) for a nonlinear state space model with additive Gaussian noise.

        x_{k+1} = transition(x_k, t_k, dt) + v_k,  v_k ~ N(0, Q)
        y_k     = observation(x_k) + w_k,            w_k ~ N(0, R)

    Improves over the bootstrap filter by pre-weighting particles using a pilot
    prediction mu = transition(x, t, dt) (no noise) before resampling, then
    correcting with the likelihood ratio p(y | x_prop) / p(y | mu_ancestor).

    Args:
        transition:    f(x, t, dt) -> x_next, supports batched x of shape (..., D)
        observation:   h(x) -> y, supports batched x of shape (..., D)
        t:             observation times, shape (T,)
        y:             observations, shape (T, O)
        n_particles:   number of particles
        x0_mean:       prior mean of initial state, shape (D,)
        x0_cov:        prior covariance of initial state, shape (D, D)
        Q:             process noise covariance, shape (D, D)
        R:             observation noise covariance, shape (O, O)
        key:           JAX random key
        n_substeps:    number of transition substeps per observation interval (default 1)
        ess_threshold: resample (second stage) when ESS falls below this fraction (default 0.5)

    Returns:
        particles:              shape (T, N, D)
        log_weights:            shape (T, N)
        ess:                    shape (T,)
        log_marginal_likelihood: scalar estimate of log p(y_{0:T-1})
    """
    D = x0_mean.shape[0]
    ess_threshold_abs = ess_threshold * n_particles

    def log_likelihood(y_t, particles):
        return multivariate_normal.logpdf(y_t, observation(particles), R)

    def normalise(lw):
        return lw - logsumexp(lw)

    def pilot_predict(p, t_start, dt_obs):
        """Deterministic multi-substep prediction (no noise)."""
        dt_sub = dt_obs / n_substeps
        t_sub = t_start + jnp.arange(n_substeps) * dt_sub

        def substep(p, t_k):
            return transition(p, t_k, dt_sub), None

        p, _ = jax.lax.scan(substep, p, t_sub)
        return p

    def propagate(p, t_start, dt_obs, key):
        dt_sub = dt_obs / n_substeps
        t_sub = t_start + jnp.arange(n_substeps) * dt_sub
        keys = jax.random.split(key, n_substeps)

        def substep(p, inputs):
            t_k, key_k = inputs
            v = jax.random.multivariate_normal(
                key_k, jnp.zeros(D), Q, shape=(n_particles,)
            )
            return transition(p, t_k, dt_sub) + v, None

        p, _ = jax.lax.scan(substep, p, (t_sub, keys))
        return p

    def systematic_resample(p, lw, key):
        cdf = jnp.cumsum(jnp.exp(lw))
        u = (
            jnp.arange(n_particles) + jax.random.uniform(key, shape=(n_particles,))
        ) / n_particles
        idx = jnp.searchsorted(cdf, u)
        return p[idx], idx

    def resample_uniform(args):
        p, lw, key = args
        p, _ = systematic_resample(p, lw, key)
        return p, jnp.full(n_particles, -jnp.log(n_particles))

    def skip_resample(args):
        p, lw, _ = args
        return p, lw

    def step(carry, inputs):
        p, lw, key = carry
        t_k, t_next, y_next = inputs
        dt = t_next - t_k

        key, key_aux, key_prop, key_res = jax.random.split(key, 4)

        # --- First stage: auxiliary resampling ---
        mu = pilot_predict(p, t_k, dt)          # (N, D) deterministic predictions
        log_g = log_likelihood(y_next, mu)       # (N,) pilot log-likelihoods

        lw_aux_unnorm = lw + log_g
        log_Z_aux = logsumexp(lw_aux_unnorm)
        lw_aux = lw_aux_unnorm - log_Z_aux       # normalized auxiliary weights

        p_resampled, ancestor_idx = systematic_resample(p, lw_aux, key_aux)
        log_g_ancestors = log_g[ancestor_idx]    # pilot likelihoods for chosen ancestors

        # --- Propagate with noise ---
        p_prop = propagate(p_resampled, t_k, dt, key_prop)

        # --- Second stage: correction weights ---
        log_p_prop = log_likelihood(y_next, p_prop)              # (N,)
        lw_corr_unnorm = log_p_prop - log_g_ancestors            # (N,)
        log_Z_corr = logsumexp(lw_corr_unnorm) - jnp.log(n_particles)
        lw = lw_corr_unnorm - logsumexp(lw_corr_unnorm)

        log_Z = log_Z_aux + log_Z_corr
        ess = ess_from_log_weights(lw)

        # Optional second-stage resampling if correction weights are uneven
        p, lw = jax.lax.cond(
            ess < ess_threshold_abs, resample_uniform, skip_resample, (p_prop, lw, key_res)
        )

        return (p, lw, key), (p, lw, ess, log_Z)

    key, key_init = jax.random.split(key)
    p0 = jax.random.multivariate_normal(key_init, x0_mean, x0_cov, shape=(n_particles,))
    log_lik_0 = log_likelihood(y[0], p0)
    log_Z_0 = logsumexp(log_lik_0) - jnp.log(n_particles)
    lw0 = normalise(log_lik_0)
    ess0 = ess_from_log_weights(lw0)

    _, (particles, log_weights, ess, log_Zs) = jax.lax.scan(
        step, (p0, lw0, key), (t[:-1], t[1:], y[1:])
    )

    particles = jnp.concatenate([p0[jnp.newaxis], particles], axis=0)
    log_weights = jnp.concatenate([lw0[jnp.newaxis], log_weights], axis=0)
    ess = jnp.concatenate([jnp.array([ess0]), ess], axis=0)
    log_marginal_likelihood = log_Z_0 + jnp.sum(log_Zs)

    return particles, log_weights, ess, log_marginal_likelihood


def ess_from_log_weights(log_weights):
    """Effective sample size from normalised log weights, shape (N,) -> scalar."""
    return jnp.exp(-logsumexp(2.0 * log_weights))


def particle_mean(particles, log_weights):
    """Weighted mean of particles.

    Args:
        particles:   shape (T, N, D)
        log_weights: shape (T, N)
    Returns:
        shape (T, D)
    """
    w = jnp.exp(log_weights)
    return jnp.einsum("tn,tnd->td", w, particles)


def particle_std(particles, log_weights):
    """Weighted standard deviation of particles.

    Args:
        particles:   shape (T, N, D)
        log_weights: shape (T, N)
    Returns:
        shape (T, D)
    """
    w = jnp.exp(log_weights)
    mean = jnp.einsum("tn,tnd->td", w, particles)
    var = jnp.einsum("tn,tnd->td", w, jnp.square(particles)) - jnp.square(mean)
    return jnp.sqrt(jnp.maximum(var, 0.0))
