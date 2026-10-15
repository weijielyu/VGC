"""Single-video VGC inference: take a corrupted mp4, produce a refined mp4.

Everything is computed on the fly:

    corrupted.mp4 ─► VAE encode      ─► corrupted_latent (1,16,T',H',W')
                  ─► SEA-RAFT        ─► raw flow (T-1,H,W,2)
                                     ─► downsample to latent res (21,th,tw,2)
                                     ─► latent_flow (1,2,T',H',W')
    caption       ─► umt5-xxl        ─► context (1,L,4096)
                                     │
                  y = cat([flow (2ch) + zeros (2ch), corrupted_latent (16ch)], dim=1)
                                     │
    noise ─► DiT(latents, y, context) × 50 steps ─► refined_latent
    refined_latent ─► VAE decode ─► refined.mp4

Automations (all optional, CLI overrides win):
  * Checkpoint: if --checkpoint is omitted, picks the stage4 with-flow or the
    stage3 no-flow checkpoint based on --no_flow.
  * Caption:    if --caption is omitted, runs Qwen3-VL-32B on the input video
    to produce a one-line scene description.
  * Flow:       (with-flow checkpoint) SEA-RAFT runs on the input unless
    --flow_npy or --no_flow is given.
  * Resolution: (--height/--width omitted) picks the token-equivalent
    resolution whose aspect ratio best matches the input video.

Usage:
    python infer_vgc.py \\
        --input_video path/to/corrupted.mp4 \\
        --output_video outputs/refined.mp4
    # uses the with-flow checkpoint + Qwen3-VL caption + auto resolution
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import av
import numpy as np
import torch
from PIL import Image

_orig_torch_load = torch.load
def _torch_load_unsafe(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_unsafe  # type: ignore[assignment]

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, model_fn_wan_video
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.utils.data import save_video


# ─────────────────────────── default paths ──────────────────────────────────

CKPT_WITH_FLOW = "./checkpoints/wan21/stage4_mixed_with_flow_final.safetensors"
CKPT_NO_FLOW   = "./checkpoints/wan21/stage3_mixed_no_flow_final.safetensors"
QWEN_VL_DEFAULT = "./wan_models_cache/Qwen3-VL-32B-Instruct"


# ─────────────────────────── resolution picking ─────────────────────────────
# Wan2.1 VAE downsamples spatially by 8. The DiT patch_size is (1, 2, 2) so the
# per-frame spatial token grid is (H/16) × (W/16). At the training resolution
# 480×832 that's 30 × 52 = 1560 spatial tokens. Other resolutions that hit the
# same spatial token count — and therefore should preserve the per-token
# attention budget the checkpoint was trained with — are the (h, w) pairs
# below. Every entry has H/16 × W/16 == 1560 and H, W both multiples of 16.
TOKEN_EQUIVALENT_RESOLUTIONS: list[tuple[int, int]] = [
    (480, 832),    # 30 × 52  (training reference; ~9:16 landscape)
    (832, 480),    # 52 × 30  (portrait flip)
    (640, 624),    # 40 × 39  (near-square, landscape)
    (624, 640),    # 39 × 40  (near-square, portrait)
    (416, 960),    # 26 × 60  (wider landscape)
    (960, 416),    # 60 × 26  (wider portrait)
    (384, 1040),   # 24 × 65
    (1040, 384),   # 65 × 24
    (320, 1248),   # 20 × 78  (very wide)
    (1248, 320),   # 78 × 20
]


def pick_token_equivalent_resolution(src_h: int, src_w: int) -> tuple[int, int]:
    """Return the (H, W) from TOKEN_EQUIVALENT_RESOLUTIONS whose aspect ratio
    is closest (in log-ratio space) to the source video's."""
    import math
    src_logr = math.log(src_w / max(1, src_h))
    best = min(
        TOKEN_EQUIVALENT_RESOLUTIONS,
        key=lambda hw: abs(math.log(hw[1] / hw[0]) - src_logr),
    )
    return best


# ─────────────────────────── I/O helpers ────────────────────────────────────

def probe_video_size(mp4_path: str) -> tuple[int, int]:
    """Return (H, W) of the first video stream in the file."""
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream.codec_context.height, stream.codec_context.width


def load_video_frames(mp4_path: str, num_frames: int) -> list[np.ndarray]:
    """Decode mp4 → list of (H, W, 3) uint8 arrays, pad/truncate to num_frames."""
    frames: list[np.ndarray] = []
    with av.open(mp4_path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise RuntimeError(f"video {mp4_path} produced 0 frames")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def resize_frames(frames: list[np.ndarray], height: int, width: int) -> np.ndarray:
    """List of arbitrary-size (H, W, 3) uint8 → (T, height, width, 3) uint8."""
    processed = [np.array(Image.fromarray(f).resize((width, height), Image.BILINEAR))
                 for f in frames]
    return np.stack(processed, axis=0)


def frames_uint8_to_tensor(frames_np: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (C, T, H, W) float in [-1, 1]."""
    v = torch.from_numpy(frames_np)
    return v.permute(3, 0, 1, 2).float() / 127.5 - 1.0


# ─────────────────────────── Qwen3-VL captioner ────────────────────────────

def qwen3_vl_caption(
    video_path: str,
    qwen_model_path: str,
    prompt: str = "Describe this video in one concise sentence, focusing on the scene and camera motion.",
    max_new_tokens: int = 96,
) -> str:
    """Run Qwen3-VL on the input video and return a one-line caption.

    Loads the model with `device_map='auto'` so it spreads across whatever
    GPUs are visible (32B ≈ 65 GB in bf16 — prefer multi-GPU or offload).
    Releases it as soon as the caption is generated.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[caption] loading Qwen3-VL from {qwen_model_path}")
    processor = AutoProcessor.from_pretrained(qwen_model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        qwen_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": os.path.abspath(video_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    caption = processor.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True,
    )[0].strip().replace("\n", " ")

    del model, processor, inputs, output_ids
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[caption] generated: {caption!r}")
    return caption


# ─────────────────────────── flow (SEA-RAFT) ────────────────────────────────

def compute_latent_flow(
    frames_np: np.ndarray,     # (T, H, W, 3) uint8 at the target (H, W)
    device: str,
    sea_raft_repo: str,
    sea_raft_url: str | None,
    sea_raft_ckpt: str | None,
    target_h: int,             # latent height (H / 8)
    target_w: int,             # latent width  (W / 8)
) -> torch.Tensor:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flow_utils import (
        load_sea_raft, compute_flow_for_video, downsample_flow_to_latent,
    )

    cfg_path = os.path.join(sea_raft_repo, "config", "eval", "spring-M.json")
    model, raft_args, InputPadder = load_sea_raft(
        sea_raft_repo, cfg_path,
        hf_url=sea_raft_url, local_ckpt=sea_raft_ckpt, device=device,
    )
    flow_raw = compute_flow_for_video(model, raft_args, InputPadder,
                                      frames_np, device)
    del model
    torch.cuda.empty_cache()

    flow_latent = downsample_flow_to_latent(flow_raw, target_h, target_w)
    flow_t = torch.from_numpy(flow_latent.astype(np.float32))  # (T', th, tw, 2)
    return flow_t.permute(3, 0, 1, 2).contiguous()             # (2, T', th, tw)


# ─────────────────────────── conditioning construction ──────────────────────

def construct_y(corrupted_latent: torch.Tensor,
                flow_latent: torch.Tensor | None) -> torch.Tensor:
    """(B, 20, T', H', W') = [flow (2) + zeros (2), corrupted_latent (16)]."""
    B, _, T, H, W = corrupted_latent.shape
    if flow_latent is not None:
        f = flow_latent.to(device=corrupted_latent.device, dtype=corrupted_latent.dtype)
        f = f.unsqueeze(0).expand(B, -1, -1, -1, -1)
        if f.shape[2:] != (T, H, W):
            raise ValueError(
                f"flow shape {tuple(f.shape)} doesn't match corrupted latent "
                f"(B, 2, {T}, {H}, {W}). Mismatched --height/--width/--num_frames."
            )
        mask_slot = torch.cat(
            [f, torch.zeros(B, 2, T, H, W, device=f.device, dtype=f.dtype)], dim=1,
        )
    else:
        mask_slot = torch.zeros(B, 4, T, H, W,
                                device=corrupted_latent.device,
                                dtype=corrupted_latent.dtype)
    return torch.cat([mask_slot, corrupted_latent], dim=1)


# ─────────────────────────── denoise + decode ──────────────────────────────

@torch.no_grad()
def denoise(dit, scheduler, shape, y, context, clip_feature, device, dtype,
            num_inference_steps=50, seed=0):
    scheduler.set_timesteps(num_inference_steps)
    torch.manual_seed(seed)
    latents = torch.randn(shape, device=device, dtype=dtype)
    for timestep in scheduler.timesteps:
        timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)
        noise_pred = model_fn_wan_video(
            dit=dit, latents=latents, timestep=timestep,
            context=context, clip_feature=clip_feature, y=y,
        )
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.no_grad()
def decode_latent_to_frames(pipe, latent, device):
    latent = latent.to(dtype=pipe.torch_dtype, device=device)
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    video = pipe.vae.decode(latent, device=device, tiled=True)[0]  # (C, T, H, W)
    v = ((video + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).cpu().numpy()                        # (T, H, W, C)
    return [Image.fromarray(v[i]) for i in range(v.shape[0])]


# ─────────────────────────── main ──────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Single-video VGC inference")
    p.add_argument("--input_video", required=True, help="Path to corrupted.mp4")
    p.add_argument("--output_video", required=True, help="Where to write the refined mp4")

    # checkpoint (auto-picked from --no_flow if omitted)
    p.add_argument("--checkpoint", default=None,
                   help=f"Path to dit.safetensors. Default: "
                        f"{CKPT_WITH_FLOW} (with flow), {CKPT_NO_FLOW} (--no_flow).")

    # caption (auto-generated via Qwen3-VL if omitted)
    p.add_argument("--caption", default=None,
                   help="Text caption. If omitted, Qwen3-VL-32B captions "
                        "the input video automatically.")
    p.add_argument("--qwen_model", default=QWEN_VL_DEFAULT,
                   help="Local dir or HF id for the Qwen3-VL captioner.")

    # resolution (auto-picked from input aspect ratio if omitted)
    p.add_argument("--height", type=int, default=None,
                   help="Target H. Default: token-equivalent of 480×832 that "
                        "best matches input aspect ratio.")
    p.add_argument("--width", type=int, default=None, help="Target W (see --height).")
    p.add_argument("--num_frames", type=int, default=81)

    # wan model dir
    p.add_argument("--model_dir",
                   default="./wan_models_cache/Wan2.1-Fun-1.3B-InP")

    # sampler
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=15)

    # flow
    p.add_argument("--no_flow", action="store_true",
                   help="Use the no-flow checkpoint and pass zeros for the flow slot.")
    p.add_argument("--flow_npy", default=None,
                   help="Precomputed raw flow .npy (T-1, H, W, 2) at the TARGET "
                        "resolution. Skips SEA-RAFT.")
    p.add_argument("--sea_raft_repo",
                   default="./wan_models_cache/SEA-RAFT")
    p.add_argument("--sea_raft_url",
                   default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    p.add_argument("--sea_raft_ckpt", default=None)

    args = p.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    # ── auto-select checkpoint based on flow mode ─────────────────────────
    if args.checkpoint is None:
        args.checkpoint = CKPT_NO_FLOW if args.no_flow else CKPT_WITH_FLOW
        print(f"[auto] checkpoint = {args.checkpoint}")

    # ── auto-select resolution from input aspect ratio ────────────────────
    src_h, src_w = probe_video_size(args.input_video)
    if args.height is None or args.width is None:
        target_h, target_w = pick_token_equivalent_resolution(src_h, src_w)
        args.height = args.height or target_h
        args.width  = args.width  or target_w
        print(f"[auto] input {src_w}x{src_h} → target {args.width}x{args.height} "
              f"(token grid {args.height // 16}×{args.width // 16} = {args.height * args.width // 256})")

    # sanity: H, W must be divisible by 16 (VAE /8 + patch /2)
    if args.height % 16 or args.width % 16:
        raise ValueError(
            f"height={args.height} and width={args.width} must both be "
            f"divisible by 16 (VAE stride 8 × patch stride 2).")

    # ── 1. Decode + resize input video ────────────────────────────────────
    print(f"[1/6] Reading {args.input_video}")
    raw_frames = load_video_frames(args.input_video, args.num_frames)
    frames_np = resize_frames(raw_frames, args.height, args.width)  # (T, H, W, 3)
    video_t = frames_uint8_to_tensor(frames_np)                      # (C, T, H, W) [-1, 1]

    lat_h, lat_w = args.height // 8, args.width // 8
    lat_t = (args.num_frames - 1) // 4 + 1

    # ── 2. Flow (optional) ─────────────────────────────────────────────────
    flow_latent = None
    if args.no_flow:
        print("[2/6] --no_flow: flow slot will be zeros.")
    elif args.flow_npy:
        print(f"[2/6] Loading precomputed flow from {args.flow_npy}")
        raw = np.load(args.flow_npy)
        from flow_utils import downsample_flow_to_latent
        fl = downsample_flow_to_latent(raw, lat_h, lat_w)
        flow_latent = torch.from_numpy(fl.astype(np.float32)).permute(3, 0, 1, 2).contiguous()
    else:
        print(f"[2/6] SEA-RAFT flow at {args.height}x{args.width}")
        flow_latent = compute_latent_flow(
            frames_np, device=device,
            sea_raft_repo=args.sea_raft_repo,
            sea_raft_url=args.sea_raft_url,
            sea_raft_ckpt=args.sea_raft_ckpt,
            target_h=lat_h, target_w=lat_w,
        )

    # ── 3. Auto-caption via Qwen3-VL (if not provided) ─────────────────────
    if args.caption is None:
        print("[3/6] No --caption: running Qwen3-VL-32B on the input video.")
        args.caption = qwen3_vl_caption(args.input_video, args.qwen_model)
    else:
        print(f"[3/6] Using provided caption: {args.caption!r}")

    # ── 4. Build WanVideoPipeline + load VGC DiT weights ───────────────────
    print("[4/6] Loading Wan2.1 pipeline (DiT + VAE + umt5)")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device=device, redirect_common_files=False,
        model_configs=[
            ModelConfig(path=os.path.join(args.model_dir, "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=os.path.join(args.model_dir, "Wan2.1_VAE.pth")),
            ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(args.model_dir, "google/umt5-xxl")),
    )
    print(f"[4/6] Loading VGC checkpoint {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)

    # silence VAE's internal tqdm
    import diffsynth.models.wan_video_vae as _vae_mod
    _vae_mod.tqdm = lambda it, *a, **kw: it

    # ── 5. VAE encode + text encode ────────────────────────────────────────
    print(f"[5/6] VAE encoding → latent ({lat_t} × {lat_h} × {lat_w})")
    pipe.dit.to("cpu"); pipe.text_encoder.to("cpu"); pipe.vae.to(device)
    torch.cuda.empty_cache()
    corr_latent = pipe.vae.encode([video_t.to(dtype=dtype, device=device)],
                                   device=device, tiled=True)[0]
    corr_latent = corr_latent.unsqueeze(0).to(dtype=dtype, device=device)

    print("[5/6] Text encoding caption")
    pipe.vae.to("cpu"); pipe.text_encoder.to(device)
    torch.cuda.empty_cache()
    if pipe.tokenizer is not None:
        pipe.tokenizer.tokenizer.pad_token = pipe.tokenizer.tokenizer.eos_token or "</s>"
    ids, mask = pipe.tokenizer(args.caption, return_mask=True, add_special_tokens=True)
    ids = ids.to(device); mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    emb = pipe.text_encoder(ids, mask)
    context = torch.stack([emb[0][:seq_lens[0]]]).to(device=device, dtype=dtype)

    # ── 6. Denoise + decode ────────────────────────────────────────────────
    print(f"[6/6] Denoising for {args.num_inference_steps} steps")
    y = construct_y(corr_latent, flow_latent)
    clip_feature = (torch.zeros(1, 257, 1280, device=device, dtype=dtype)
                    if pipe.dit.require_clip_embedding else None)

    pipe.text_encoder.to("cpu"); pipe.dit.to(device=device, dtype=dtype); pipe.dit.eval()
    torch.cuda.empty_cache()

    refined_latent = denoise(
        pipe.dit, FlowMatchScheduler(),
        shape=(1, 16, lat_t, lat_h, lat_w),
        y=y, context=context, clip_feature=clip_feature,
        device=device, dtype=dtype,
        num_inference_steps=args.num_inference_steps, seed=args.seed,
    )

    print(f"[6/6] VAE decoding → {args.output_video}")
    pipe.dit.to("cpu"); pipe.vae.to(device)
    torch.cuda.empty_cache()
    refined_frames = decode_latent_to_frames(pipe, refined_latent.squeeze(0), device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)) or ".", exist_ok=True)
    save_video(refined_frames, args.output_video, fps=args.fps, quality=5)
    print(f"Done: {args.output_video}")


if __name__ == "__main__":
    main()
