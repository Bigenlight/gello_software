#!/usr/bin/env python3
"""JPEG -> policy-ready image tensor, byte-parity with training / eval_offline.py.

The trained checkpoint declares its cameras at 3x720x1280 (config.json), but the
policy was trained on 360x640 crops: training passed
`--dataset.image_transforms.tfs='{"resize":{"weight":1.0,"type":"Resize",
"kwargs":{"size":[360,640]}}}'`, and eval_offline.build_image_transforms()
replicates that with a single deterministic torchvision `v2.Resize(size=[360,640])`.

Crucially, the SAVED policy preprocessor (policy_preprocessor.json) does NOT resize
and does NOT convert colour order -- it only renames/batches/moves-to-device and
normalizes with the baked-in mean/std. So the resize + BGR->RGB MUST happen HERE,
before the saved preprocessor runs, exactly as the dataset did before training.

Parity chain vs eval_offline.py:
  * eval_offline: LeRobotDataset decodes video -> float32 CHW in [0,1] -> the same
    `v2.Resize(size=[360,640])` (dataset_reader.get_item applies image_transforms to
    the already-float frame; verified in lerobot 0.6.1).
  * here:         cv2.imdecode(jpeg) -> BGR uint8 HWC -> RGB -> float32 CHW in [0,1]
                  -> the same `v2.Resize(size=[360,640])`.
  Both feed the Resize an identical float32 CHW [0,1] RGB tensor, so the output is
  byte-identical (same op, same torchvision version). Verified in the smoke test.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

# Training / eval resolution (H, W). MUST match eval_offline.RESIZE_HW.
RESIZE_HW = [360, 640]

# Single deterministic Resize, constructed EXACTLY like eval_offline's
# build_image_transforms() (lerobot make_transform_from_config does
# `getattr(v2, "Resize")(size=[360,640])` with no extra kwargs). torchvision v2
# defaults are InterpolationMode.BILINEAR + antialias=True -> "bilinear + antialias".
_RESIZE = v2.Resize(size=RESIZE_HW)


def decode_jpeg_to_rgb_float_chw(jpeg_bytes: bytes) -> torch.Tensor:
    """Decode one JPEG frame into a policy-ready image tensor.

    Args:
        jpeg_bytes: raw JPEG payload (the ROS CompressedImage `msg.data`), as sent
            unmodified over ZMQ by the py3.10 node.

    Returns:
        torch.float32 tensor of shape (3, 360, 640), CHW, RGB, values in [0, 1],
        resized with the same transform used at training time.

    Raises:
        ValueError: if the bytes cannot be decoded as an image.
    """
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    # IMREAD_COLOR -> 3-channel BGR uint8 HWC (drops alpha, forces 3ch).
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed: not a valid JPEG/image payload")

    # BGR -> RGB to match the dataset (RealSense frames were stored/decoded as RGB).
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # HWC uint8 [0,255] -> CHW float32 [0,1]. This mirrors the dataset's float frame
    # that the Resize saw at training time (float BEFORE resize -> byte parity).
    t = torch.from_numpy(np.ascontiguousarray(rgb))  # (H, W, 3) uint8
    t = t.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)  # (3, H, W)

    # Same torchvision Resize as eval_offline -> (3, 360, 640) float32 [0,1].
    return _RESIZE(t)
