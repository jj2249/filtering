import jax
import jax.numpy as jnp


def simulate(transition, observation, x0, T, dt_sim, dt_obs, Q, R, key):
    """
    Simulate a nonlinear state space model with additive Gaussian noise.

        x_{k+1} = transition(x_k, t_k, dt_sim) + v_k,  v_k ~ N(0, Q)
        y_k     = observation(x_k) + w_k,               w_k ~ N(0, R)

    Args:
        transition: f(x, t, dt) -> x_next, supports batched x of shape (..., D)
        observation: h(x) -> y, supports batched x of shape (..., D)
        x0:     initial state, shape (D,)
        T:      end time
        dt_sim: simulation time step
        dt_obs: observation interval (must be an integer multiple of dt_sim)
        Q:      process noise covariance, shape (D, D)
        R:      observation noise covariance, shape (O, O)
        key:    JAX random key

    Returns:
        (t_sim, x_sim): times (N_sim+1,) and states (N_sim+1, D)
        (t_obs, y_obs): times (N_obs+1,) and observations (N_obs+1, O)
    """
    sample_every_float = dt_obs / dt_sim
    sample_every = int(round(sample_every_float))
    assert jnp.isclose(jnp.array(sample_every_float), jnp.array(float(sample_every))), (
        "dt_obs must be an integer multiple of dt_sim"
    )

    D = Q.shape[0]
    O = R.shape[0]
    N_sim = int(round(T / dt_sim))

    t_sim = jnp.arange(N_sim + 1) * dt_sim

    key, key_v, key_w = jax.random.split(key, 3)

    v_seq = jax.random.multivariate_normal(key_v, jnp.zeros(D), Q, shape=(N_sim,))

    def step(x, inputs):
        t_k, v_k = inputs
        return transition(x, t_k, dt_sim) + v_k, x

    x_final, x_seq = jax.lax.scan(step, x0, (t_sim[:-1], v_seq))
    x_sim = jnp.concatenate([x_seq, x_final[jnp.newaxis]], axis=0)

    sample_indices = jnp.arange(0, N_sim + 1, sample_every)
    t_obs = t_sim[sample_indices]
    x_obs = x_sim[sample_indices]

    N_obs = sample_indices.shape[0]
    w_seq = jax.random.multivariate_normal(key_w, jnp.zeros(O), R, shape=(N_obs,))
    y_obs = observation(x_obs) + w_seq

    return (t_sim, x_sim), (t_obs, y_obs)
