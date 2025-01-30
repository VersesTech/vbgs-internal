# Copyright 2024 VERSES AI, Inc.
#
# Licensed under the VERSES Academic Research License (the “License”);
# you may not use this file except in compliance with the license.
#
# You may obtain a copy of the License at
#
#     https://github.com/VersesTech/vbgs/blob/main/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import jax
import jax.numpy as jnp

import equinox as eqx

from vbgs.model.train import compute_elbo_delta


@jax.jit
def update_initial_model(initial_model, s_means, c_means):
    initial_model = eqx.tree_at(
        lambda model: model.likelihood._posterior_params.eta.eta_1,
        initial_model,
        replace=s_means * initial_model.likelihood.kappa,
    )
    initial_model = eqx.tree_at(
        lambda model: model.delta._posterior_params.eta.eta_1,
        initial_model,
        replace=c_means * initial_model.delta.kappa,
    )
    return initial_model

def padzero(data, batch_size):
    datalen = len(data)
    n_batches = datalen // batch_size
    remainder = datalen - (n_batches * batch_size)
    topad = batch_size - remainder
    return jnp.concatenate((data, jnp.zeros((topad, *data.shape[1:]))), axis=0), topad


def reassign2(
    initial_model, model, data, batch_size, fraction=0.05, debug=False
):
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    padded_data, cutoff = padzero(data, batch_size)

    def step(xi):
        elbos, post = compute_elbo_delta(initial_model, xi)
        return elbos

    padded_data = jnp.reshape(padded_data, (n_batches, batch_size, *data.shape[1:], 1))
    elbos = jax.lax.map(step, padded_data).flatten()
    p_elbo = -elbos[:-cutoff]
    # p_elbo_min = -jnp.nanmax(elbos.at[:-cutoff].set(jnp.nan))  # using this way the array shape should vary less
    
    # This sum is still quite heavy in the profile.
    available = jnp.sum(
        model.prior.alpha <= initial_model.prior.prior_alpha.min()
    )

    n_reassign = int(available * fraction)

    # p_elbo = -elbos
    p_elbo = p_elbo - p_elbo.min()  # smallest value 0
    p_elbo = p_elbo / p_elbo.sum()  # sum to 1

    point_idcs = np.random.choice(
        np.arange(len(p_elbo)),
        p=p_elbo,
        size=n_reassign,
        replace=False,
    )
    component_idcs = model.prior.alpha.argsort()[:n_reassign]  # this is also not going to be very fast..

    # basically, if we can set the means of the initial model to these data
    # points, we can do a regular update after.
    s_means = initial_model.likelihood.mean
    s_means = s_means.at[component_idcs].set(data[point_idcs, :3, jnp.newaxis])

    c_means = initial_model.delta.mean
    c_means = c_means.at[component_idcs].set(data[point_idcs, 3:, jnp.newaxis])

    initial_model = update_initial_model(initial_model, s_means, c_means)

    if debug:
        # plot_selection(elbos, point_idcs, data[:, 3:].reshape((512, 512, 3)))
        return initial_model, {"elbo": elbos, "p_elbo": p_elbo}
    else:
        return initial_model



def reassign(
    initial_model, model, data, batch_size, fraction=0.05, debug=False
):
    """
    Heuristic to force better assignments. Takes n points with the lowest elbo,
    and reassigns them to n components that are currently unused.
    n is determined dynamically as a fraction of the unused components.
    """

    n_batches = int(np.ceil(data.shape[0] / batch_size))
    elbos = jnp.zeros((0))
    for batch_idx in range(n_batches):
        xi = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        xi = jnp.expand_dims(jnp.array(xi), -1)

        size = xi.shape[0]
        if size < batch_size:
            # Concat zeros, so that the posteriors are still computed with jit
            # NOTE: the elbo will not be correct since it's contributing the
            # log likelihood of the augmented zeros. But since we don't use the
            # elbo here, it is not a problem.
            xi = jnp.concatenate(
                [xi, jnp.zeros((batch_size - size, *xi.shape[1:]))],
                axis=0,
            )
            elbo, posteriors = compute_elbo_delta(initial_model, xi)
            xi = xi[:size]
            posteriors = posteriors[:size]
            elbo = elbo[:size]
        else:
            elbo, posteriors = compute_elbo_delta(initial_model, xi)

        elbos = jnp.concatenate([elbos, elbo], axis=0)

    available = jnp.sum(
        model.prior.alpha <= initial_model.prior.prior_alpha.min()
    )

    n_reassign = int(available * fraction)

    p_elbo = -elbos
    p_elbo = p_elbo - p_elbo.min()  # smallest value 0
    p_elbo = p_elbo / p_elbo.sum()  # sum to 1

    point_idcs = np.random.choice(
        np.arange(len(elbos)),
        p=p_elbo,
        size=n_reassign,
        replace=False,
    )
    component_idcs = model.prior.alpha.argsort()[:n_reassign]

    # basically, if we can set the means of the initial model to these data
    # points, we can do a regular update after.
    s_means = initial_model.likelihood.mean
    s_means = s_means.at[component_idcs].set(data[point_idcs, :3, jnp.newaxis])

    c_means = initial_model.delta.mean
    c_means = c_means.at[component_idcs].set(data[point_idcs, 3:, jnp.newaxis])

    initial_model = update_initial_model(initial_model, s_means, c_means)

    if debug:
        # plot_selection(elbos, point_idcs, data[:, 3:].reshape((512, 512, 3)))
        return initial_model, {"elbo": elbos, "p_elbo": p_elbo}
    else:
        return initial_model
