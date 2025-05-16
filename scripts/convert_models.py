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

import pickle
import json
import jax.numpy as jnp

import os
from pathlib import Path
import tqdm

import numpy as np

if __name__ == "__main__":
    # Converts models from json to npz (less storage required)
    p = "/home/toon.vandemaele/projects/iclr-rebuttal/vbgs-internal/scripts/data/sweep/"
    p = Path(p)

    files = list(p.glob("**/model_*.json"))
    print(files)
    for f in tqdm.tqdm(files):
        pa = str(f).replace(".json", ".npz")
        if os.path.exists(pa):
            continue

        with open(f, "r") as fp:
            d = json.load(fp)

        mu = np.asarray(d["mu"])
        si = np.asarray(d["si"])
        alpha = np.asarray(d["alpha"])

        with open(pa, "wb") as fp:
            np.savez(fp, mu=mu, si=si, alpha=alpha)

        del d
        os.remove(str(f))
