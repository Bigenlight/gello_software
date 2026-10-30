#!/usr/bin/env python3
"""JPEG -> policy-ready image tensor for the FLOW-MATCHING (multi_task_dit) deploy.

FM variant of image_preprocess.py. The ONE difference from the diffusion path is
that this module does NOT externally resize the frame:

  * The diffusion checkpoint was trained on 360x640 crops and its saved
    preprocessor does NOT resize, so image_preprocess.py had to pre-resize to
    360x640 before the saved preprocessor ran.
  * The flow-matching multi_task_dit checkpoint resizes INTERNALLY: its RGB
    encoder applies `torchvision.Resize(config.image_resize_shape)` (= [224, 224])
    on every forward (modeling_multi_task_dit.py, the image encoder's do_resize
    branch). Pre-resizing here to some other size would double-resize and break
    training parity. So we feed the frame at its NATIVE decode resolution and let
    the policy resize to 224x224 itself -- exactly as the dataset did at training
    time (frames stored native, resized to 224 inside the model).

Everything else matches the diffusion path (and the dataset decode):
  cv2.imdecode(jpeg) -> BGR uint8 HWC -> RGB -> float32 CHW in [0, 1].
The saved FM preprocessor (policy_preprocessor.json) then rename/batch/tokenize/
device/normalize (VISUAL = MEAN_STD) -- it does NOT resize and does NOT convert
colour order, so the BGR->RGB MUST happen HERE, before it runs.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# No RESIZE_HW / no v2.Resize here on purpose: the FM policy resizes to
# config.image_resize_shape (= [224, 224]) internally. See module docstring.


def decode_jpeg_to_rgb_float_chw(jpeg_bytes: bytes) -> torch.Tensor:
    """Decode one JPEG frame into a policy-ready image tensor (NO external resize).

    Args:
        jpeg_bytes: raw JPEG payload (the ROS CompressedImage `msg.data`), as sent
            unmodified over ZMQ by the py3.10 node.

    Returns:
        torch.float32 tensor of shape (3, H, W), CHW, RGB, values in [0, 1], at the
        frame's NATIVE decode resolution. The FM policy resizes it to 224x224
        internally (config.image_resize_shape).

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
    # the policy saw at training time (the internal Resize sees an identical float32
    # CHW [0,1] RGB tensor -> parity). NO external resize -- the policy does it.
    t = torch.from_numpy(np.ascontiguousarray(rgb))  # (H, W, 3) uint8
    t = t.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)  # (3, H, W)
    return t
