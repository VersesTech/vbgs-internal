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

from PIL import Image
import numpy as np

from pathlib import Path

import sys
import vbgs


def load_depth_model(model_name, device):
    import torch

    if model_name == "zoe":
        return load_zoe(device)
    elif model_name == "dpt":
        return load_dpt(device)
    else:
        return load_dav2(device)


def load_dpt(device):
    model_type = "DPT_Large"
    # model_type = "DPT_Hybrid"
    # model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if "small" in model_type:
        depth_transforms = midas_transforms.dpt_transform
    else:
        depth_transforms = midas_transforms.small_transform
    depth_model = midas

    def preprocess(x):
        return depth_transforms(x).to(device)

    def post_process(x, shape):
        x = torch.nn.functional.interpolate(
            x.unsqueeze(1),
            size=shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        scale = 0.000305
        shift = 0.1378

        depth = scale * x + shift
        depth = 1 / depth.cpu().detach().numpy()
        depth[depth < 1e-8] = 1e-8
        return depth

    return depth_model, preprocess, post_process


def load_zoe(device):
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe = zoe.to(device)

    def model(img):
        return zoe.infer_pil(
            Image.fromarray(img.astype(np.uint8)), output_type="tensor"
        ).unsqueeze(0)

    def post_process(x, shape):
        x = torch.nn.functional.interpolate(
            x.unsqueeze(1),
            size=shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return x.detach().numpy()

    return model, lambda x: x, post_process


def load_dav2(device):
    root_path = Path(vbgs.__file__).parent.parent
    da_path = root_path / "../Depth-Anything-V2"
    sys.path.append(str((da_path / "metric_depth").absolute()))

    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    encoder = "vitl"  # or 'vits', 'vitb'
    # 'hypersim' for indoor model, 'vkitti' for outdoor model
    dataset = "hypersim"
    # 20 for indoor model, 80 for outdoor model
    max_depth = 20 if dataset == "hypersim" else 80

    model = DepthAnythingV2(
        **{**model_configs[encoder], "max_depth": max_depth}
    )
    model.load_state_dict(
        torch.load(
            da_path
            / f"checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth",
            map_location="cpu",
            weights_only=True,
        )
    )

    model = model.to(device).eval()

    def preprocess(x):
        # model takes BGR
        return x[..., [2, 1, 0]]

    def postprocess(x, shape):
        return x * 0.7

    return model.infer_image, preprocess, postprocess


def predict_depth(rgb, depth_model, preprocess, postprocess):
    x = preprocess(rgb)

    with torch.no_grad():
        prediction = depth_model(x)

    depth = postprocess(prediction, x.shape)

    return depth
