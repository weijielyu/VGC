"""Batch VGC inference using Wan2.2-I2V-A14B (MoE: high-noise + low-noise
experts). Walks an input directory of mp4s, refines each, writes refined
mp4 to an output dir. Caption auto-generated via Qwen3-VL-32B (or supplied
via --captions_json).

Pipeline per video:
    input.mp4 ─► uniform-sample 81 frames ─► resize to 720×1248
              ─► VAE encode → corrupted_latent
    caption   ─► umt5-xxl    → context
              y = [zeros(4ch), corrupted_latent(16ch)]   # no_flow
    noise + y + context ─► DiT_high (steps 0-17) → DiT_low (steps 17-50)
                       ─► refined_latent ─► VAE decode → refined.mp4

Usage (single GPU, walks a dir):
    python infer_vgc_wan22.py \\
        --input_dir  inputs \\
        --output_dir outputs_wan22

Usage (8-GPU sharded — videos split across ranks):
    torchrun --standalone --nproc_per_node 8 infer_vgc_wan22.py \\
        --input_dir  inputs \\
        --output_dir outputs_wan22

Captions: by default each video is auto-captioned by Qwen3-VL-32B at the
start of its turn. To skip Qwen, write a JSON `{ "<basename>": "caption" }`
and pass --captions_json.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
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


# ─────────────────────────── defaults (Wan2.2-I2V-A14B) ─────────────────────

CKPT_HIGH = "./checkpoints/wan22/release_no_flow_high_noise.pth"
CKPT_LOW  = "./checkpoints/wan22/release_no_flow_low_noise.pth"
WAN22_DIR = "./wan_models_cache/Wan2.2-I2V-A14B"
QWEN_VL_DEFAULT = "./wan_models_cache/Qwen3-VL-32B-Instruct"

# Wan2.2 14B was fine-tuned at 720×1248. The DiT patch + VAE produce a token
# grid of (1248/16) × (720/16) = 78 × 45 = 3510 spatial tokens. We don't try
# to support other aspect ratios here — the training is fixed at this shape.
TARGET_H, TARGET_W = 720, 1248
NUM_FRAMES = 81
LATENT_T = (NUM_FRAMES - 1) // 4 + 1   # 21
LATENT_H, LATENT_W = TARGET_H // 8, TARGET_W // 8  # 90, 156


# ─────────────────────────── I/O helpers ────────────────────────────────────

def probe_video(mp4_path: str) -> tuple[int, int, int]:
    """Return (H, W, total_frames) of the first video stream."""
    with av.open(mp4_path) as container:
        s = container.streams.video[0]
        n = s.frames
        if n == 0:  # some encoders don't fill nb_frames
            for _ in container.decode(video=0): n += 1
        return s.codec_context.height, s.codec_context.width, n


def load_uniform_frames(mp4_path: str, num_frames: int) -> list[np.ndarray]:
    """Decode mp4 and uniformly sample `num_frames` frames across the whole
    clip. Falls back to first-N + last-frame padding if the video is shorter."""
    _, _, total = probe_video(mp4_path)
    if total <= 0:
        raise RuntimeError(f"video {mp4_path} has 0 frames")
    indices = np.linspace(0, max(0, total - 1), num_frames).round().astype(int)
    target_set = sorted(set(indices.tolist()))

    decoded: dict[int, np.ndarray] = {}
    with av.open(mp4_path) as container:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx in target_set:
                decoded[frame_idx] = frame.to_ndarray(format="rgb24")
            if frame_idx > target_set[-1]:
                break

    if not decoded:
        raise RuntimeError(f"video {mp4_path} produced 0 sampled frames")
    last = sorted(decoded.keys())[-1]
    out = [decoded.get(i, decoded[last]) for i in indices.tolist()]
    return out


def resize_frames(frames: list[np.ndarray], h: int, w: int) -> np.ndarray:
    return np.stack(
        [np.array(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in frames],
        axis=0,
    )


def frames_uint8_to_tensor(frames_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 127.5 - 1.0


# ─────────────────────────── Qwen3-VL captioner ────────────────────────────

class QwenCaptioner:
    """Lazily loads Qwen3-VL-32B once and reuses for all videos in this rank.
    Holds the model on device_map='auto' (~65 GB bf16); call .release() when
    done so we can swap the GPU back to Wan2.2 inference."""
    def __init__(self, qwen_model_path: str, max_new_tokens: int = 96):
        self.qwen_model_path = qwen_model_path
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print(f"[caption] loading Qwen3-VL from {self.qwen_model_path}")
        self.processor = AutoProcessor.from_pretrained(self.qwen_model_path,
                                                       trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        ).eval()

    def caption(self, video_path: str,
                prompt: str = ("Describe this video in one concise sentence, "
                               "focusing on the scene and camera motion.")) -> str:
        self._ensure_loaded()
        messages = [{"role": "user", "content": [
            {"type": "video", "video": os.path.abspath(video_path)},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                             do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True,
        )[0].strip().replace("\n", " ")
        del inputs, output_ids
        return text

    def release(self):
        if self.model is None: return
        del self.model, self.processor
        self.model = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()


# ─────────────────────────── conditioning ───────────────────────────────────

def construct_y(corrupted_latent: torch.Tensor) -> torch.Tensor:
    """y = [zeros(4ch), corrupted_latent(16ch)] (no_flow case)."""
    B, _, T, H, W = corrupted_latent.shape
    mask_slot = torch.zeros(B, 4, T, H, W,
                            device=corrupted_latent.device,
                            dtype=corrupted_latent.dtype)
    return torch.cat([mask_slot, corrupted_latent], dim=1)


# ─────────────────────────── block offload (≤48 GB GPUs) ────────────────────

def enable_block_offload(dit, num_blocks: int, device: str):
    """Keep the last `num_blocks` transformer blocks on CPU and move each one
    to `device` only for its own forward call. One 14B expert (28 GB) plus the
    720p activations does not fit in 48 GB; this trades speed for peak VRAM.
    Placement only — the computation is unchanged."""
    blocks = list(dit.blocks)[-num_blocks:] if num_blocks > 0 else []
    for block in blocks:
        block.register_forward_pre_hook(lambda m, args: m.to(device) and None)
        block.register_forward_hook(lambda m, args, out: m.to("cpu") and None)
    dit._vgc_offload_blocks = blocks


def _to_gpu(dit, device: str):
    """Move an expert to `device`, leaving its offloaded blocks on CPU."""
    dit.to(device=device)
    for block in getattr(dit, "_vgc_offload_blocks", []):
        block.to("cpu")


# ─────────────────────────── two-expert denoise ─────────────────────────────

@torch.no_grad()
def denoise_moe(dit_high, dit_low, scheduler, shape, y, context, clip_feature,
                device, dtype, num_inference_steps=50, seed=0,
                switch_DiT_boundary=0.358):
    """Step-INDEX-based switch: first `boundary*num_steps` steps use the
    high-noise expert (the same scheduler-index split the experts were trained on).

    Memory: only ONE 14B expert is resident on GPU at a time. dit_high stays
    on GPU through the high-noise phase, then we swap (high→CPU, low→GPU)
    exactly once at the boundary step. With 720×1248×81 (~3500 spatial tokens
    per frame × 21 frames) the activation budget can hit ~45 GB on its own,
    so we can't afford to keep both 14B experts (56 GB) in VRAM concurrently.
    """
    scheduler.set_timesteps(num_inference_steps)
    torch.manual_seed(seed)
    latents = torch.randn(shape, device=device, dtype=dtype)
    boundary_step = int(switch_DiT_boundary * num_inference_steps)

    # Initial state: high on GPU (caller guarantees), low on CPU.
    dit_low.to("cpu")
    _to_gpu(dit_high, device)
    torch.cuda.empty_cache()
    swapped = False

    for step_idx, timestep in enumerate(scheduler.timesteps):
        if step_idx == boundary_step and not swapped:
            dit_high.to("cpu")
            torch.cuda.empty_cache()
            _to_gpu(dit_low, device)
            swapped = True
        active_dit = dit_low if swapped else dit_high
        timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)
        noise_pred = model_fn_wan_video(
            dit=active_dit, latents=latents, timestep=timestep,
            context=context, clip_feature=clip_feature, y=y,
        )
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


@torch.no_grad()
def decode_latent_to_frames(pipe, latent, device):
    latent = latent.to(dtype=pipe.torch_dtype, device=device)
    if latent.dim() == 4:
        latent = latent.unsqueeze(0)
    video = pipe.vae.decode(latent, device=device, tiled=True)[0]
    v = ((video + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).cpu().numpy()
    return [Image.fromarray(v[i]) for i in range(v.shape[0])]


# ─────────────────────────── pipeline build ─────────────────────────────────

def build_pipeline_with_vgc_weights(model_dir: str, ckpt_high: str, ckpt_low: str,
                                    device: str, dtype: torch.dtype):
    high = sorted(glob.glob(os.path.join(model_dir, "high_noise_model",
                                         "diffusion_pytorch_model*.safetensors")))
    low = sorted(glob.glob(os.path.join(model_dir, "low_noise_model",
                                        "diffusion_pytorch_model*.safetensors")))
    if not high or not low:
        raise FileNotFoundError(f"Wan2.2 expert shards not found under {model_dir}")

    # Load on CPU: the two 14B experts (2 × 28 GB in bf16) never fit on one GPU
    # together. Each sub-model is moved to `device` only while it is in use.
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype, device="cpu", redirect_common_files=False,
        model_configs=[
            ModelConfig(path=high),
            ModelConfig(path=low),
            ModelConfig(path=os.path.join(model_dir, "Wan2.1_VAE.pth")),
            ModelConfig(path=os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(model_dir, "google/umt5-xxl")),
    )
    if pipe.dit2 is None:
        raise RuntimeError("pipe.dit2 is None — Wan2.2 base didn't load both experts.")

    print(f"[load] VGC high → pipe.dit  ({ckpt_high})")
    pipe.dit.load_state_dict(torch.load(ckpt_high, map_location="cpu", mmap=True),
                             strict=True)
    print(f"[load] VGC low  → pipe.dit2 ({ckpt_low})")
    pipe.dit2.load_state_dict(torch.load(ckpt_low, map_location="cpu", mmap=True),
                              strict=True)
    pipe.dit.to(dtype=dtype).eval()
    pipe.dit2.to(dtype=dtype).eval()

    # silence VAE's tiled-decode tqdm
    import diffsynth.models.wan_video_vae as _vae_mod
    _vae_mod.tqdm = lambda it, *a, **kw: it
    return pipe


# ─────────────────────────── per-video refine ───────────────────────────────

def refine_one(pipe, scheduler, captioner, video_path: str, out_path: str,
               caption: str | None, args, device: str, dtype: torch.dtype):
    print(f"\n=== {os.path.basename(video_path)} ===")

    # 1. decode + resize + sample
    print(f"[1/5] sampling {NUM_FRAMES} frames uniformly + resize to "
          f"{TARGET_H}×{TARGET_W}")
    raw = load_uniform_frames(video_path, NUM_FRAMES)
    frames_np = resize_frames(raw, TARGET_H, TARGET_W)
    video_t = frames_uint8_to_tensor(frames_np)

    # 2. caption
    if caption is None:
        print("[2/5] Qwen3-VL caption")
        caption = captioner.caption(video_path)
    else:
        print(f"[2/5] using supplied caption: {caption!r}")

    # 3. VAE encode + text encode (DiTs to CPU during this)
    print("[3/5] VAE encode")
    pipe.dit.to("cpu"); pipe.dit2.to("cpu"); pipe.text_encoder.to("cpu")
    pipe.vae.to(device); torch.cuda.empty_cache()
    corr_latent = pipe.vae.encode([video_t.to(dtype=dtype, device=device)],
                                  device=device, tiled=True)[0]
    corr_latent = corr_latent.unsqueeze(0).to(dtype=dtype, device=device)

    print("[3/5] umt5 text encode")
    pipe.vae.to("cpu"); pipe.text_encoder.to(device); torch.cuda.empty_cache()
    if pipe.tokenizer is not None:
        pipe.tokenizer.tokenizer.pad_token = pipe.tokenizer.tokenizer.eos_token or "</s>"
    ids, mask = pipe.tokenizer(caption, return_mask=True, add_special_tokens=True)
    ids = ids.to(device); mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    emb = pipe.text_encoder(ids, mask)
    context = torch.stack([emb[0][:seq_lens[0]]]).to(device=device, dtype=dtype)

    # 4. denoise (denoise_moe places one expert on GPU at a time)
    print(f"[4/5] denoise {args.num_inference_steps} steps "
          f"(boundary @ step {int(args.switch_DiT_boundary * args.num_inference_steps)})")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    y = construct_y(corr_latent)
    clip_feature = (torch.zeros(1, 257, 1280, device=device, dtype=dtype)
                    if pipe.dit.require_clip_embedding else None)
    refined = denoise_moe(
        pipe.dit, pipe.dit2, scheduler,
        shape=(1, 16, LATENT_T, LATENT_H, LATENT_W),
        y=y, context=context, clip_feature=clip_feature,
        device=device, dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed, switch_DiT_boundary=args.switch_DiT_boundary,
    )

    # 5. decode
    print("[5/5] VAE decode → mp4")
    pipe.dit.to("cpu"); pipe.dit2.to("cpu"); pipe.vae.to(device)
    torch.cuda.empty_cache()
    refined_frames = decode_latent_to_frames(pipe, refined.squeeze(0), device)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    save_video(refined_frames, out_path, fps=args.fps, quality=5)

    # free the GPU for the next iter (experts stay on CPU until denoise)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"     wrote {out_path}")


# ─────────────────────────── main ──────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Batch VGC-Wan2.2 mp4 refinement")
    p.add_argument("--input_dir", required=True,
                   help="Dir of input mp4s (one per scene)")
    p.add_argument("--output_dir", required=True,
                   help="Output dir; refined_<basename>.mp4 per input")
    p.add_argument("--captions_json", default=None,
                   help='JSON {"<basename_without_ext>": "caption"}. '
                        "Missing entries fall back to Qwen3-VL-32B autocaption.")
    p.add_argument("--high_checkpoint", default=CKPT_HIGH)
    p.add_argument("--low_checkpoint",  default=CKPT_LOW)
    p.add_argument("--model_dir", default=WAN22_DIR)
    p.add_argument("--qwen_model", default=QWEN_VL_DEFAULT)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--switch_DiT_boundary", type=float, default=0.358)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=15)
    p.add_argument("--offload_blocks", type=int, default=0,
                   help="Keep the last N DiT blocks of each expert on CPU, moving "
                        "each to GPU only for its forward. Use ~12 on 48 GB GPUs "
                        "(slower); 0 keeps the whole expert on GPU (needs ~80 GB).")
    args = p.parse_args()

    # multi-GPU sharding via torchrun
    use_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size = 0, 1
        device = "cuda"
    dtype = torch.bfloat16

    # collect input mp4s
    mp4s = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    if rank == 0:
        print(f"Found {len(mp4s)} mp4s in {args.input_dir}")
    my_mp4s = mp4s[rank::world_size]
    if rank == 0 and use_distributed:
        print(f"Sharded across {world_size} ranks; this rank: {len(my_mp4s)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1 (rank 0 only): caption every input mp4 with Qwen3-VL-32B,
    # write to a shared JSON. Other ranks barrier-wait. We do this BEFORE
    # loading Wan2.2 so Qwen's ~65 GB bf16 footprint can use the full GPU
    # without fighting two 28 GB DiTs.
    captions_path = os.path.join(args.output_dir, "_captions.json")
    captions: dict[str, str] = {}

    if args.captions_json:
        with open(args.captions_json) as f:
            captions = json.load(f)
        if rank == 0:
            print(f"[caption] using supplied {args.captions_json}: {len(captions)} entries")
    elif rank == 0:
        # Qwen3-VL uses device_map='auto' to spread across all visible GPUs.
        if os.path.exists(captions_path):
            with open(captions_path) as f:
                captions = json.load(f)
            print(f"[caption] resume from {captions_path}: {len(captions)} cached")
        captioner = QwenCaptioner(args.qwen_model)
        for mp4 in mp4s:
            base = os.path.splitext(os.path.basename(mp4))[0]
            if base in captions and captions[base]:
                continue
            try:
                cap = captioner.caption(mp4)
            except Exception as e:
                print(f"[caption] FAIL {base}: {e!r}", file=sys.stderr)
                cap = "A high-quality video."
            captions[base] = cap
            # ascii-safe print: stdout may be in C/POSIX locale (no UTF-8).
            print("[caption] " + base + ": " + cap[:120].encode("ascii", "replace").decode())
            with open(captions_path, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
        captioner.release()
        torch.cuda.empty_cache()
        print(f"[caption] wrote {captions_path} ({len(captions)} entries)")

    if use_distributed:
        import torch.distributed as dist
        dist.barrier()
    if rank != 0 and not args.captions_json:
        with open(captions_path) as f:
            captions = json.load(f)

    # ── Phase 2: each rank loads Wan2.2 and processes its shard.
    pipe = build_pipeline_with_vgc_weights(
        args.model_dir, args.high_checkpoint, args.low_checkpoint, device, dtype,
    )
    if args.offload_blocks > 0:
        enable_block_offload(pipe.dit, args.offload_blocks, device)
        enable_block_offload(pipe.dit2, args.offload_blocks, device)
    scheduler = FlowMatchScheduler()
    # Loaded lazily, only for videos missing from --captions_json.
    fallback_captioner = QwenCaptioner(args.qwen_model)

    for mp4 in my_mp4s:
        base = os.path.splitext(os.path.basename(mp4))[0]
        out_path = os.path.join(args.output_dir, f"refined_{base}.mp4")
        if os.path.exists(out_path):
            print(f"skip {base}: {out_path} exists")
            continue
        try:
            refine_one(pipe, scheduler, fallback_captioner, mp4, out_path,
                       caption=captions.get(base), args=args,
                       device=device, dtype=dtype)
        except Exception as e:
            print(f"FAIL {base}: {e!r}", file=sys.stderr)

    fallback_captioner.release()
    if use_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
