from functools import partial

import matplotlib.pyplot as plt

from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import jax.random as jr
import jaxsplat

from vbgs.render.volume import rot_mat_to_quat, opengl_to_colmap_frame
from vbgs.metrics import calc_psnr


@partial(jax.jit, static_argnames=["h", "w", "c", "f"])
def train_render(mu, scale, mat_lower, alpha, camera, h, w, c, f):
    rotation = jax.vmap(lambda x: x @ x.T)(mat_lower)
    wxyz = jax.vmap(rot_mat_to_quat)(rotation)
    w2c = jnp.linalg.inv(opengl_to_colmap_frame(camera))
    return jaxsplat.render(
        mu[:, :3].astype(jnp.float32),
        scale.astype(jnp.float32),
        wxyz.astype(jnp.float32),
        mu[:, 3:].astype(jnp.float32),
        alpha.astype(jnp.float32),
        viewmat=w2c.astype(jnp.float32),
        background=jnp.zeros(3, dtype=jnp.float32),
        img_shape=(h, w),
        glob_scale=1.0,
        c=c,
        f=f,
        clip_thresh=0.01,
        block_size=16,
    )


def finetune(mu, si, alpha, iterator_fn, n_iters, key, bs=64, log_every=10):
    key, subkey = jr.split(key)
    data_iter = iterator_fn(subkey)
    alpha_n = (alpha[..., None] > 0.01).astype(jnp.float64)
    mat_lower = jax.vmap(jnp.linalg.cholesky)(si[:, :3, :3])
    scales = jax.vmap(lambda x: jnp.linalg.norm(x, axis=-1))(mat_lower)
    mat_lower = mat_lower / jnp.expand_dims(scales, -1)
    params = (mu, scales, mat_lower)
    key, subkey = jr.split(key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    render_fn = partial(
        train_render, h=data_iter.h, w=data_iter.w, c=data_iter.c, f=data_iter.f
    )

    def loss_fn(mean, scale, mat_low, alpha, x, cam):
        x_hat = jax.lax.map(lambda c: render_fn(mean, scale, mat_low, alpha, c), cam)
        return jnp.mean(jnp.square(x_hat - x[..., :3] / 255))

    grad_fn = jax.value_and_grad(loss_fn, [0, 1, 2, 3])
    for i in tqdm(range(n_iters)):
        key, subkey = jr.split(key)
        batched_iter = Batched(iterator_fn(key), bs=bs)
        losses = []

        for xs, cams in tqdm(batched_iter, leave=False, total=len(batched_iter)):
            loss, grads = grad_fn(*params, alpha_n, xs, cams)
            losses.append(loss)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(grads, updates)

        if i % log_every == 0:
            img_hat = jnp.clip(render_fn(*params, alpha_n, cams[0]), min=0.0, max=1.0)
            img = xs[0][..., :3] / 255
            se = jnp.square(img_hat - img)
            _, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(img)
            axes[1].imshow(img_hat)
            axes[2].imshow(se)
            [a.axis("off") for a in axes.flatten()]
            plt.savefig(f"output_{i}.png")
            plt.close()
            tqdm.write(f"Avg mse train loss: {jnp.mean(jnp.array(losses))}")
            tqdm.write(f"Test image psnr: {calc_psnr(img, img_hat)}")

    si = si.at[:, :3, :3].set(scales * jnp.vmap(lambda x: x @ x.T)(mat_lower))
    return params[0], si, alpha


class Batched:
    """Batched loader

    For internal use specifically for loading batched sequences of frames and
    camera poses.
    """

    def __init__(self, data_loader, bs=64):
        self.i = 0
        self.inner_loader = data_loader
        self.bs = bs

    def __next__(self):
        cams, xs = [], []
        while i < len(self.inner_loader):
            for _ in range(self.bs):

                if i >= len(self.inner_loader):
                    raise StopIteration

                cams.append(self.inner_loader.load_camera_params(i)[0])
                xs.append(self.inner_loader.get_camera_frame(i)[0])
                i += 1
            cams = jnp.stack(cams)
            xs = jnp.stack(xs)
            return xs, cams

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.inner_loader) // self.bs
