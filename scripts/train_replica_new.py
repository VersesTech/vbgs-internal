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

import hydra
from omegaconf import DictConfig, OmegaConf

import json
import copy
import rich

from pathlib import Path
from tqdm import tqdm

import numpy as np
import random

import jax
import jax.numpy as jnp
import jax.random as jr

import vbgs
from vbgs.data.utils import create_normalizing_params, normalize_data
from vbgs.model.utils import random_mean_init, store_model
from vbgs.model.train import fit_gmm_step

from vbgs.data.replica import ReplicaDataIterator
from vbgs.model.reassign import reassign


from model_volume import get_volume_delta_mixture

from vbgs.render.volume import render_gsplat

from PIL import Image
from vbgs.metrics import calc_psnr, calc_mse
import matplotlib.pyplot as plt

from tqdm import trange


def evaluate(splat, cameras, eval_frames, intrinsics, shape, store_path):
    psnrs = []
    mses = []
    for i in range(len(cameras)):
        x = np.array(Image.open(eval_frames[i])) / 255.0
        c = int(intrinsics[0, 2]), int(intrinsics[1, 2])
        f = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        x_hat = render_gsplat(*splat, cameras[i], c, f, *shape)

        psnrs.append(calc_psnr(x, x_hat))
        mses.append(calc_mse(x, x_hat))

    np.savez(store_path, psnr=np.array(psnrs), mse=np.array(mses))
    return np.array(psnrs), np.array(mses)


def fit_continual(
    data_path, n_components, key=None, eval_every=1, batch_size=5000, seed=0
):
    np.random.seed(seed)
    random.seed(seed)

    if key is None:
        key = jr.PRNGKey(0)

    # Some subsampling
    data_iter = ReplicaDataIterator(data_path, None, subsample=100_000)
    eval_cameras = [
        data_iter.load_camera_params(i)[1]
        for i in jnp.arange(0, len(data_iter), 2)
    ]
    eval_frames = [
        data_iter.get_frame(i)[0] for i in jnp.arange(0, len(data_iter), 2)
    ]
    data_iter.indices = np.arange(0, 2000, 400)

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=None,
        component_shape=(n_components,),
        event_shape=(6, 1),
        init_random=True,
        add_noise=True,
    )

    key, subkey = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=subkey,
        n_components=n_components,
        mean_init=mean_init,
        beta=0,
        learning_rate=1,
        dof_offset=1,
        position_scale=n_components,
        position_event_shape=(3, 1),
    )

    data = jnp.array([data_iter.get(i) for i in range(len(data_iter))])

    def _fit_step_fn(carry, n):
        x = data[n]

        model, prior_stats, space_stats, color_stats = fit_gmm_step(
            prior_model,
            carry["model"],
            data=x[::4],
            batch_size=batch_size,
            prior_stats=carry["prior_stats"],
            space_stats=carry["space_stats"],
            color_stats=carry["color_stats"],
        )

        return {
            "model": model,
            "prior_stats": prior_stats,
            "space_stats": space_stats,
            "color_stats": color_stats,
        }, None

    # Step 0
    carry, _ = _fit_step_fn(
        {
            "model": copy.deepcopy(prior_model),
            "prior_stats": None,
            "space_stats": None,
            "color_stats": None,
        },
        0,
    )

    metrics = dict(
        {"psnr": {"mean": [], "std": []}, "mse": {"mean": [], "std": []}}
    )
    for step in trange(1, len(data_iter), eval_every):
        carry, _ = jax.lax.scan(
            _fit_step_fn,
            carry,
            jnp.arange(step * eval_every, (step + 1) * eval_every),
        )

        p, m = evaluate(
            carry["model"].extract_model(None),
            eval_cameras,
            eval_frames,
            data_iter.intrinsics,
            (data_iter.h, data_iter.w),
            f"results_{step:02d}.npz",
        )

        metrics["psnr"]["mean"].append(p.mean())
        metrics["psnr"]["std"].append(p.std())
        metrics["mse"]["mean"].append(m.mean())
        metrics["mse"]["std"].append(m.std())
        print(f"PSNR: {p.mean():.2f} +- {p.std():.2f}")


def run_experiment(
    key,
    data_path,
    n_components,
    batch_size,
    device,
):
    # Fit continual VBEM
    key, subkey = jr.split(key)
    metrics = fit_continual(
        data_path,
        n_components,
        key=subkey,
        batch_size=batch_size,
    )
    rich.print(metrics)

    return metrics


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="replica",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    jax.config.update("jax_default_device", jax.devices()[int(cfg.device)])

    root_path = Path(vbgs.__file__).parent.parent

    # Minor hack to launch everything at once
    data_path = cfg.data.data_path
    if "room0_depth_estimate" in data_path:
        data_path = data_path.replace("_depth_estimated", "").replace(
            "Replica", "Replica-depth_estimated"
        )

    results = run_experiment(
        key=jr.PRNGKey(0),
        n_components=cfg.model.n_components,
        data_path=root_path / Path(data_path),
        batch_size=cfg.train.batch_size,
        device=cfg.device,
    )
    results.update({"config": OmegaConf.to_container(cfg)})

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
