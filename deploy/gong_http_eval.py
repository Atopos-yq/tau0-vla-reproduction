"""HTTP open-loop evaluation with Tensor and image layout conversion."""
from __future__ import annotations

import numpy as np
import torch

from deploy import openloop_with_server as evaluator
from deploy.wire import pack_payload


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    return value


def image_to_hwc_uint8(value, *, name):
    image = np.asarray(to_numpy(value))

    if image.ndim != 3:
        raise ValueError(f"{name}: expected rank-3 image, got {image.shape}")

    if isinstance(value, torch.Tensor):
        # LeRobot image tensors use CHW.
        if image.shape[0] not in (1, 3):
            raise ValueError(f"{name}: invalid CHW tensor: {image.shape}")
        image = image.transpose(1, 2, 0)
    elif image.shape[-1] not in (1, 3):
        if image.shape[0] not in (1, 3):
            raise ValueError(f"{name}: invalid image layout: {image.shape}")
        image = image.transpose(1, 2, 0)

    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{name}: image contains NaN or Inf")
        if image.min() < 0 or image.max() > 1:
            raise ValueError(
                f"{name}: float image must be in [0,1], "
                f"got [{image.min()}, {image.max()}]"
            )
        image = np.rint(image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f"{name}: unsupported image dtype: {image.dtype}")

    return np.ascontiguousarray(image)


def pack_observation(payload):
    converted = to_numpy(payload)
    converted["images"] = {
        name: image_to_hwc_uint8(image, name=name)
        for name, image in payload["images"].items()
    }
    return pack_payload(converted)


def main():
    # Reuse the official CLI, dataset pipeline, action merging and metrics.
    evaluator.pack_payload = pack_observation
    evaluator.main()


if __name__ == "__main__":
    main()
