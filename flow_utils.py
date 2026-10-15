"""SEA-RAFT optical flow helpers for infer_vgc.py.

load_sea_raft builds the model from a SEA-RAFT clone, compute_flow_for_video
runs it on consecutive frame pairs, and downsample_flow_to_latent brings the
flow to the latent grid used as the DiT's flow conditioning.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _build_raft_args(cfg_path: str, overrides: dict) -> SimpleNamespace:
    """Construct a SimpleNamespace from a SEA-RAFT eval JSON config plus a
    few CLI overrides. The config files mirror the ones in
    SEA-RAFT/config/eval/, e.g. spring-M.json."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def load_sea_raft(
    repo_dir: str,
    cfg_path: str,
    *,
    hf_url: Optional[str],
    local_ckpt: Optional[str],
    device: str,
    iters_override: Optional[int] = None,
):
    """Import SEA-RAFT from its repo, build the model, load weights, return
    (model, args). The repo's `core/` directory holds raft.py with relative
    imports (`from update import ...`), so we must put `core/` itself on
    sys.path — not its parent. We undo the sys.path mutation on the way out
    so the rest of the program stays clean."""
    core_dir = os.path.join(repo_dir, "core")
    if not os.path.isdir(core_dir):
        raise FileNotFoundError(
            f"SEA-RAFT core dir not found at {core_dir}. "
            f"Expected the SEA-RAFT repo to be cloned at {repo_dir}.")

    sys.path.insert(0, core_dir)
    try:
        from raft import RAFT
        from utils.utils import InputPadder  # type: ignore
    finally:
        sys.path.remove(core_dir)

    overrides: dict = {}
    if iters_override is not None:
        overrides["iters"] = int(iters_override)
    args = _build_raft_args(cfg_path, overrides)

    if hf_url:
        model = RAFT.from_pretrained(hf_url, args=args)
    elif local_ckpt:
        model = RAFT(args)
        sd = torch.load(local_ckpt, map_location="cpu")
        # SEA-RAFT checkpoints are often wrapped in {"model": state_dict}.
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        # Strip "module." DDP prefix if present.
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[load_sea_raft] missing={len(missing)} unexpected={len(unexpected)}",
                  file=sys.stderr)
    else:
        raise ValueError("provide either --sea_raft_url or --sea_raft_ckpt")

    model = model.to(device).eval()
    return model, args, InputPadder


@torch.no_grad()
def compute_flow_for_video(
    model,
    args,
    InputPadder,
    frames_np: np.ndarray,    # (T, H, W, 3) uint8
    device: str,
) -> np.ndarray:
    """Run SEA-RAFT on every consecutive frame pair of one video. Returns
    a (T-1, H, W, 2) float16 numpy array of forward flow vectors in
    pixel units at the ORIGINAL frame resolution."""
    T, H, W, _ = frames_np.shape

    # (T, 3, H, W) float32 in [0, 255] — SEA-RAFT expects float in 0..255.
    frames_t = torch.from_numpy(frames_np).to(device, dtype=torch.float32)
    frames_t = frames_t.permute(0, 3, 1, 2).contiguous()

    # Optional scale factor from config (spring-M uses scale=-1, i.e.
    # downsample inputs by 0.5 then multiply output flow by 0.5).
    scale = float(getattr(args, "scale", 0))
    scale_factor = 2 ** scale
    if scale != 0:
        frames_t = F.interpolate(
            frames_t,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )

    padder = InputPadder(frames_t.shape, mode="kitti")
    frames_t_padded = padder.pad(frames_t)[0]

    out = np.empty((T - 1, H, W, 2), dtype=np.float16)
    iters = int(getattr(args, "iters", 4))

    for i in range(T - 1):
        img1 = frames_t_padded[i:i + 1]
        img2 = frames_t_padded[i + 1:i + 2]
        result = model(img1, img2, iters=iters, test_mode=True)
        # SEA-RAFT returns {'flow': [list across iterations], 'info': [...]}
        flow_list = result["flow"]
        flow = flow_list[-1] if isinstance(flow_list, (list, tuple)) else flow_list
        flow = padder.unpad(flow)              # (1, 2, h_scaled, w_scaled)

        if scale != 0:
            # Upsample flow back to original input resolution AND rescale
            # magnitudes (a 1-pixel displacement at half resolution = 2-pixel
            # displacement at full resolution, hence the multiply by 1/scale_factor).
            flow = F.interpolate(flow, size=(H, W), mode="bilinear",
                                 align_corners=False)
            flow = flow * (1.0 / scale_factor)

        # (1, 2, H, W) → (H, W, 2)
        flow_np = flow[0].permute(1, 2, 0).contiguous().to(torch.float32).cpu().numpy()
        out[i] = flow_np.astype(np.float16)

    return out


def downsample_flow_to_latent(
    flow: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Downsample raw flow (80, H, W, 2) to latent-resolution (21, th, tw, 2).

    Spatial: bilinear resize + magnitude rescale.
    Temporal: average 4 consecutive flows → 20 positions + 1 zero = 21.
    """
    T, H, W, _ = flow.shape  # (80, 480, 832, 2)

    # Spatial downsample
    flow_t = torch.from_numpy(flow.astype(np.float32))  # (80, H, W, 2)
    flow_t = flow_t.permute(0, 3, 1, 2)  # (80, 2, H, W)
    flow_r = F.interpolate(flow_t, size=(target_h, target_w),
                           mode="bilinear", align_corners=False)
    # Rescale magnitudes
    flow_r[:, 0, :, :] *= target_w / W  # dx
    flow_r[:, 1, :, :] *= target_h / H  # dy
    flow_r = flow_r.permute(0, 2, 3, 1)  # (80, th, tw, 2)

    # Temporal: group 4 consecutive → average → 20 frames + 1 zero
    n_groups = T // 4  # 80 // 4 = 20
    grouped = flow_r[:n_groups * 4].reshape(n_groups, 4, target_h, target_w, 2)
    averaged = grouped.mean(dim=1)  # (20, th, tw, 2)

    # Last position = zeros (frame 80 has no forward flow)
    zero = torch.zeros(1, target_h, target_w, 2)
    result = torch.cat([averaged, zero], dim=0)  # (21, th, tw, 2)

    return result.numpy().astype(np.float16)

