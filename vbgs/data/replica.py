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

import json
from pathlib import Path
from PIL import Image

import jax
import jax.numpy as jnp
import jax.random as jr


from functools import partial

from vbgs.camera import transform_uvd_to_points
from vbgs.data.utils import normalize_data


depth_scale = 6553.5
fx, fy, x0, y0 = 600.0, 600.0, 599.5, 339.5
h, w = 680, 1200
intrinsics = jnp.array([
    [fx, 0.0, x0, 0.0],
    [0.0, fy, y0, 0.0],
    [0.0, 0.0 , 1.0, 0.0],
    [0.0 ,0.0 , 0.0, 1.0]
])

class ReplicaDataIterator:
    def __init__(
        self,
        datapath,
        data_params=None,
        subsample=None
    ):
        self._data_params = data_params
        self._subsample = subsample
        self._datapath = Path(datapath) / "results/"
        posepath = Path(datapath) / "traj.txt"
        with open(posepath, "r") as f:
            lines = f.readlines()
            lines = jnp.array(list(map(
                lambda l: [float (x) for x in l.strip().split(" ")],
                lines
            )))
        poses = jnp.reshape(lines, (len(lines), 4, 4))
        opengl_to_frame = jnp.array(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]]
        )
        f = jax.vmap(lambda x: jnp.dot(x, opengl_to_frame))
        self.poses = f(poses)
        self.i = 0
        self.key = jr.PRNGKey(0)


    def __len__(self):
        return len(self.poses)

    def __iter__(self):
        return self

    def __next__(self):
        # Couldn't find the parameters in a config file (I think they are
        # in the gradslam repo instead). So I just copied them using the
        # debugger.
        if self.i >= len(self):
            raise StopIteration
        idx = str(self.i).zfill(6)
        color = jnp.array(Image.open(self._datapath / f"frame{idx}.jpg").resize((w, h), Image.Resampling.NEAREST))
        depth = jnp.array(Image.open(self._datapath / f"depth{idx}.png").resize((w, h), Image.Resampling.NEAREST))
        depth = depth / depth_scale
        camera_to_world = self.poses[self.i]
        self.i += 1
        data = jnp.concatenate(
            transform_uvd_to_points(
                color[..., :3],
                depth,
                camera_to_world,
                intrinsics,
                from_opengl=True,
                filter_zero=True
            ),
            axis=1
        )
        if self._data_params is not None:
            data, _ = normalize_data(data, self._data_params)
        if self._subsample is not None:
            self.key, subkey = jr.split(self.key)
            data = jr.permutation(subkey, data, independent=False)
            data = data[:self._subsample]
        return data