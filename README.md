# Learning to Correct Geometry in Generated Videos

### 🦘 NeurIPS 2026 🍑

[Weijie Lyu](https://weijielyu.github.io/)<sup>1</sup>, [Tiancheng Shen](https://scholar.google.com/citations?user=iRY1YVoAAAAJ&hl=en)<sup>1</sup>, [Xiangtai Li](https://lxtgh.github.io/)<sup>2</sup>, [Yujing Wang](https://scholar.google.com/citations?user=YgL4rywAAAAJ&hl=en)<sup>3</sup>, [Xiaoyuan Wang](https://www.linkedin.com/in/xiaoyuan-isaac-wang/)<sup>4</sup>, [Haobo Yuan](https://yuanhaobo.me/)<sup>1</sup>, [Yizhou Zhao](https://yizhou42.github.io/)<sup>4</sup>, [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<sup>1</sup>  
<sup>1</sup>UC Merced &nbsp;·&nbsp; <sup>2</sup>NTU &nbsp;·&nbsp; <sup>3</sup>PKU &nbsp;·&nbsp; <sup>4</sup>CMU

[![Website](https://img.shields.io/badge/Website-VGC?logo=googlechrome&logoColor=hsl(204%2C%2086%25%2C%2053%25)&label=VGC&labelColor=%23f5f5dc&color=hsl(204%2C%2086%25%2C%2053%25))](https://weijielyu.github.io/VGC)
[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)](https://arxiv.org/abs/XXXX.XXXXX)
[![Models](https://img.shields.io/badge/Models-HuggingFace?logo=huggingface&logoColor=%23FFD21E&label=HuggingFace&labelColor=%23f5f5dc&color=%23FFD21E)](https://huggingface.co/wlyu/VGC)

<div align='center'>
<img alt="teaser" src='media/teaser.jpg'>
</div>

> *Video Geometry Corrector (VGC)* is a post-hoc video-to-video model that corrects geometry distortions in generated videos — warped structures, inconsistent parallax, drifting scene layouts — without modifying the source generator. Scaled to a larger backbone, it extends to general video refinement of dynamic human and object motion.

## 🔧 Prerequisites

### Environment Setup

```bash
conda create -n vgc python=3.11 -y
conda activate vgc

# Install PyTorch matching your CUDA, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the dependencies (this also pulls the DiffSynth-Studio backbone)
pip install -e .

# Optional: flash-attention for faster inference (falls back to PyTorch SDPA otherwise)
pip install flash-attn --no-build-isolation
```

### Downloads

We release three models. Download the base weights for the model(s) you want to run:

| Model | Resolution | Base model | VGC checkpoint | Optical flow |
| --- | --- | --- | --- | --- |
| Wan 2.1 1.3B (default) | 480p | Wan2.1-Fun-1.3B-InP | `wan21/stage4_mixed_with_flow_final` | ✓ |
| Wan 2.1 1.3B, no flow | 480p | Wan2.1-Fun-1.3B-InP | `wan21/stage3_mixed_no_flow_final` | — |
| Wan 2.2 14B MoE | 720p | Wan2.2-I2V-A14B | `wan22/no_flow_high_step4500` + `wan22/no_flow_low_step3500` | — |

**1. Base model weights** (via Hugging Face):

```bash
pip install huggingface_hub

# Wan2.1-Fun-1.3B-InP (~15 GB; the CLIP image encoder is not needed)
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
  --include "diffusion_pytorch_model.safetensors" --include "Wan2.1_VAE.pth" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" --include "config.json" --include "configuration.json" \
  --include "google/umt5-xxl/*" \
  --local-dir ./wan_models_cache/Wan2.1-Fun-1.3B-InP

# Wan2.2-I2V-A14B (~126 GB), only for the 14B model
hf download Wan-AI/Wan2.2-I2V-A14B \
  --include "high_noise_model/*" --include "low_noise_model/*" --include "Wan2.1_VAE.pth" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" --include "configuration.json" --include "google/umt5-xxl/*" \
  --local-dir ./wan_models_cache/Wan2.2-I2V-A14B
```

**2. VGC checkpoints** from [Hugging Face](https://huggingface.co/wlyu/VGC):

```bash
hf download wlyu/VGC --local-dir ./checkpoints
```

**3. Optical flow and captioning** — [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) computes the
optical flow for the default model, and [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
captions the input video:

```bash
git clone https://github.com/princeton-vl/SEA-RAFT ./wan_models_cache/SEA-RAFT   # weights auto-download from HF
pip install scipy opencv-python-headless h5py                                    # SEA-RAFT dependencies
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir ./wan_models_cache/Qwen3-VL-32B-Instruct
```

Both are optional: SEA-RAFT is not used by the no-flow models, and Qwen3-VL is skipped when you
provide the caption yourself (`--caption` / `--captions_json`). Any Qwen3-VL size works via
`--qwen_model`; the default 32B needs ~65 GB in bf16.

The expected layout:

```
wan_models_cache/
├── Wan2.1-Fun-1.3B-InP/
│   ├── diffusion_pytorch_model.safetensors
│   ├── Wan2.1_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/
├── Wan2.2-I2V-A14B/
│   ├── high_noise_model/
│   ├── low_noise_model/
│   ├── Wan2.1_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/
├── SEA-RAFT/
└── Qwen3-VL-32B-Instruct/

checkpoints/
├── wan21/
│   ├── stage4_mixed_with_flow_final.safetensors
│   └── stage3_mixed_no_flow_final.safetensors
└── wan22/
    ├── no_flow_high_step4500.safetensors
    └── no_flow_low_step3500.safetensors
```

## 🚀 Inference

VGC takes a geometry-distorted video and outputs a geometry-corrected one. Sample distorted
videos are provided in [`inputs/`](inputs/).

### Wan 2.1 (1.3B, 480p)

```bash
# Default model, with optical flow
python infer_vgc.py \
    --input_video  inputs/000.mp4 \
    --output_video outputs/000.mp4

# No-flow model
python infer_vgc.py --no_flow \
    --input_video  inputs/000.mp4 \
    --output_video outputs/000.mp4
```

The output resolution is the 480p-equivalent (same token count as 480×832) that best matches the
input aspect ratio. To process a directory:

```bash
for f in inputs/*.mp4; do
    python infer_vgc.py --input_video "$f" --output_video "outputs/$(basename "$f")"
done
```

**Available Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--input_video` | — | Input distorted video |
| `--output_video` | — | Output corrected video |
| `--no_flow` | off | Use the no-flow model |
| `--checkpoint` | `checkpoints/wan21/...` | VGC checkpoint (picked from `--no_flow` if omitted) |
| `--caption` | Qwen3-VL | Text caption of the input video |
| `--height`, `--width` | auto | Output resolution (multiples of 16) |
| `--num_frames` | `81` | Number of leading frames read from the input (padded with the last frame if shorter) |
| `--num_inference_steps` | `50` | Number of denoising steps |
| `--seed` | `0` | Random seed |
| `--fps` | `15` | Output frame rate |

### Wan 2.2 (14B MoE, 720p)

The 14B model samples 81 frames uniformly from each input and outputs 720×1248, so it also performs
480p→720p super-resolution. It processes every `.mp4` in `--input_dir` and writes
`refined_<name>.mp4` to `--output_dir`.

**Single GPU**

```bash
python infer_vgc_wan22.py \
    --input_dir  inputs \
    --output_dir outputs_wan22
```

**Multi-GPU** — videos are distributed across GPUs:

```bash
torchrun --standalone --nproc_per_node 8 infer_vgc_wan22.py \
    --input_dir  inputs \
    --output_dir outputs_wan22
```

**Low-VRAM Mode** — one 14B expert at 720p needs ~80 GB. On GPUs with less memory (e.g. 48 GB),
keep the last DiT blocks on the CPU and move each to the GPU only for its own forward pass:

```bash
python infer_vgc_wan22.py \
    --input_dir  inputs \
    --output_dir outputs_wan22 \
    --offload_blocks 12
```

This trades speed for memory; the results are the same.

Captions are generated by Qwen3-VL and cached in `<output_dir>/_captions.json`. To provide your own,
pass `--captions_json captions.json` with `{"<video name without .mp4>": "caption", ...}`.

## 📝 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{lyu2026vgc,
  title     = {Learning to Correct Geometry in Generated Videos},
  author    = {Lyu, Weijie and Shen, Tiancheng and Li, Xiangtai and Wang, Yujing and
               Wang, Xiaoyuan and Yuan, Haobo and Zhao, Yizhou and Yang, Ming-Hsuan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## 📄 License

Codes are licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Acknowledgements

This work is built upon [Wan](https://arxiv.org/abs/2503.20314) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
Optical flow uses [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) and captions use [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL). We thank the authors for their excellent work.
