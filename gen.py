"""
Memory-managed Krea-2 Turbo runner. Supports two checkpoint formats and an
optional LoRA, and is tuned to fit a 24 GB GPU.

Key idea: the Qwen text encoder (~8 GB) is only needed to encode the prompt
once, up front. So we encode first, evict the encoder from the GPU, and only
then put the MMDiT on the GPU for the denoise loop. That frees ~8 GB for the
diffusion model and avoids running at the VRAM ceiling.

Two checkpoint formats are auto-detected:
  * fp8 krea2_turbo_fp8.safetensors (~12 GB, float8_e4m3fn) -> stays fp8 and
    resident on the GPU; weights are dequantized per-layer just before each
    matmul. No RAM streaming, so it is fast.
  * bf16 turbo.safetensors (~26 GB) -> too big to sit on a 24 GB card, so the
    MMDiT is streamed from system RAM with accelerate.cpu_offload.

Checkpoint path comes from --ckpt (preferred) or the OSS_TURBO env var. An
optional --lora <file.safetensors> is merged into the base weights at load.
CLI mirrors inference.py so the GUI can drive it.
"""

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # offload hooks vs torch.compile
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import sys
import types
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE  # gen.py now lives inside the krea-2 repo
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # so the cloned repo's modules import cleanly

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from accelerate import cpu_offload

from autoencoder import QwenAutoencoder
from encoder import Qwen3VLConditioner, TextEncoderConfig
import mmdit
from mmdit import SingleMMDiTConfig, SingleStreamDiT
from sampling import sample

from einops import rearrange


def _attention_any(q, k, v, mask=None, scale=None, gqa=False):
    """Upstream mmdit forces SDPBackend.CUDNN_ATTENTION, which reports
    'No available kernel' for these shapes on some torch/CUDA builds (e.g.
    torch 2.11 + cu128). Let SDPA pick a working backend instead."""
    x = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _attention_any  # Attention.forward looks this up in mmdit's globals

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

SINGLE_MMDIT_LARGE_WIDE = SingleMMDiTConfig(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
    multiplier=4, layers=28, patch=2, channels=16,
    txtheads=20, txtkvheads=20, txtlayers=12,
)
TEXT_ENCODER = TextEncoderConfig(model_id="Qwen/Qwen3-VL-4B-Instruct")


class _CachedEncoder:
    """Stand-in passed to sample(): returns text embeddings encoded earlier,
    so the real encoder can be evicted from the GPU before the denoise loop."""
    def __init__(self, cache):
        self._cache = cache  # tuple(prompts) -> (txt, txtmask)

    def __call__(self, prompts):
        return self._cache[tuple(prompts)]


def _is_fp8(state):
    return any(v.dtype in FP8_DTYPES for v in state.values())


def _patch_fp8_linears(model):
    """Make every nn.Linear dequantize an fp8 weight to the activation dtype
    just before the matmul, so fp8 weights can stay resident on the GPU."""
    def forward(self, x):
        w = self.weight
        if w.dtype in FP8_DTYPES:
            w = w.to(x.dtype)
        b = self.bias
        if b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        return F.linear(x, w, b)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            m.forward = types.MethodType(forward, m)


def _merge_lora(state, lora_path, user_scale):
    """Best-effort LoRA merge into the base state dict (in place).

    Handles common naming conventions (lora_A/lora_B, lora_down/lora_up,
    optional .alpha, and kohya-style lora_unet_ prefixes with underscores).
    Each match adds (alpha/rank) * scale * (up @ down) to the base weight.
    """
    lora = load_file(lora_path)
    bases = {k[:-7] for k in state if k.endswith(".weight")}  # 'blocks.0.attn.wq'
    # kohya replaces dots with underscores in names, which is ambiguous for
    # modules like 'txtfusion.layerwise_blocks' that already contain an
    # underscore. Match against the real names' underscore form to resolve it.
    und = {b.replace(".", "_"): b for b in bases}

    pairs = {}
    suffixes = [(".lora_A.weight", "down"), (".lora_B.weight", "up"),
                (".lora_down.weight", "down"), (".lora_up.weight", "up"),
                (".lora_A", "down"), (".lora_B", "up")]
    for k, v in lora.items():
        hit = False
        for suf, role in suffixes:
            if k.endswith(suf):
                pairs.setdefault(k[:-len(suf)], {})[role] = v
                hit = True
                break
        if not hit and k.endswith(".alpha"):
            pairs.setdefault(k[:-6], {})["alpha"] = float(v.flatten()[0])

    def normalize(b):
        b = b.replace("lora_unet_", "").replace("lora_te_", "")
        for pre in ("diffusion_model.", "model.diffusion_model.",
                    "transformer.", "model."):
            if b.startswith(pre):
                b = b[len(pre):]
        return b

    matched = 0
    unmatched = []
    for base, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            continue
        nb = normalize(base)
        target = next((c for c in (nb, nb.replace("_", ".")) if c in bases), None)
        if target is None:
            target = und.get(nb)  # resolve kohya underscore names exactly
        if target is None:
            unmatched.append(base)
            continue
        down = parts["down"].to(torch.float32)
        up = parts["up"].to(torch.float32)
        rank = down.shape[0]
        scale = (parts.get("alpha", rank) / rank) * user_scale
        delta = (up @ down) * scale
        wname = target + ".weight"
        w = state[wname]
        if delta.shape != tuple(w.shape):
            unmatched.append(base)
            continue
        state[wname] = (w.to(torch.float32) + delta).to(w.dtype)  # keeps fp8 fp8
        matched += 1

    print(f"[lora] merged {matched} modules into the base weights "
          f"(unmatched: {len(unmatched)})", flush=True)
    if matched == 0:
        print("[lora] WARNING: nothing matched. Sample LoRA keys:",
              list(lora.keys())[:6], flush=True)
    return matched


def _encode(prompts, negatives, device, dtype):
    """Load the encoder on the GPU, encode, then evict it. Returns a cache and
    a freshly-freed GPU."""
    encoder = Qwen3VLConditioner(
        TEXT_ENCODER.model_id, TEXT_ENCODER.max_length,
        select_layers=TEXT_ENCODER.select_layers,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    cache = {}
    with torch.no_grad():
        cache[tuple(prompts)] = encoder(prompts)
        if negatives is not None:
            cache[tuple(negatives)] = encoder(negatives)
    encoder.to("cpu")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return cache


def _load_mmdit(ckpt_path, lora, lora_scale, device, dtype):
    with torch.device("meta"):
        mmdit = SingleStreamDiT(SINGLE_MMDIT_LARGE_WIDE)

    state = load_file(ckpt_path)
    fp8 = _is_fp8(state)

    # Some community checkpoints (e.g. the FP8 export) ship a few extra tensors
    # this architecture does not use (last.down / last.up). Drop anything the
    # model does not expect so a strict load still succeeds.
    expected = set(mmdit.state_dict().keys())
    extra = [k for k in list(state) if k not in expected]
    for k in extra:
        state.pop(k)
    if extra:
        print(f"[load] ignored {len(extra)} unused key(s): {extra}", flush=True)

    if lora:
        _merge_lora(state, lora, lora_scale)

    mmdit.load_state_dict(state, strict=True, assign=True)
    mmdit = mmdit.eval().requires_grad_(False)
    del state

    if fp8:
        for p in mmdit.parameters():  # keep fp8 fp8; cast tiny params to bf16
            if p.dtype not in FP8_DTYPES and p.is_floating_point():
                p.data = p.data.to(dtype)
        _patch_fp8_linears(mmdit)
        try:
            return mmdit.to(device), "fp8-resident"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return cpu_offload(mmdit, execution_device=torch.device(device)), "fp8-offload"

    mmdit = mmdit.to(dtype)
    return cpu_offload(mmdit, execution_device=torch.device(device)), "bf16-offload"


def run(prompt, ckpt, steps, cfg, mu, width, height, num_images, seed,
        output, lora=None, lora_scale=1.0, device="cuda", dtype=torch.bfloat16):
    prompts = [prompt] * num_images
    negatives = [""] * num_images if cfg > 0 else None

    print("Encoding prompt (first run downloads the Qwen text encoder ~8 GB - "
          "one-time)...", flush=True)
    cache = _encode(prompts, negatives, device, dtype)

    print(f"Loading {os.path.basename(ckpt)}...", flush=True)
    mmdit, mode = _load_mmdit(ckpt, lora, lora_scale, device, dtype)
    ae = QwenAutoencoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    print(f"Loaded. VRAM strategy: {mode}", flush=True)

    print(f"Generating {num_images} image(s) at {width}x{height}, {steps} "
          "steps...", flush=True)
    images = sample(
        mmdit, ae, _CachedEncoder(cache), prompts,
        negative_prompts=negatives, width=width, height=height, steps=steps,
        guidance=cfg, seed=seed, mu=mu,
    )
    for i, image in enumerate(images):
        out = f"{output}_{i}.png"
        image.save(out)
        print(f"saved {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Krea-2 Turbo (memory-managed).")
    ap.add_argument("prompt")
    ap.add_argument("--ckpt", default=os.environ.get("OSS_TURBO"),
                    help="path to a turbo .safetensors (bf16 or fp8)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=1.15)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--num-images", dest="num_images", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="sample")
    ap.add_argument("--lora", default=None, help="optional LoRA .safetensors")
    ap.add_argument("--lora-scale", dest="lora_scale", type=float, default=1.0)
    # Depth ControlNet-LoRA mode (image-guided). Both enable it together.
    ap.add_argument("--depth-image", dest="depth_image", default=None,
                    help="input image for depth ControlNet-LoRA mode")
    ap.add_argument("--depth-lora", dest="depth_lora", default=None,
                    help="depth-control LoRA .safetensors (needs the bf16 base)")
    args = ap.parse_args()

    if not args.ckpt or not os.path.isfile(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}", flush=True)
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available.", flush=True)
        return 1
    if args.lora and not os.path.isfile(args.lora):
        print(f"ERROR: LoRA file not found: {args.lora}", flush=True)
        return 1

    # Depth ControlNet-LoRA mode: image-guided generation via depth_control.py.
    # Needs an input image, the depth LoRA, and the BF16 base (turbo.safetensors).
    if args.depth_image or args.depth_lora:
        if not (args.depth_image and os.path.isfile(args.depth_image)):
            print(f"ERROR: depth input image not found: {args.depth_image}", flush=True)
            return 1
        if not (args.depth_lora and os.path.isfile(args.depth_lora)):
            print(f"ERROR: depth LoRA not found: {args.depth_lora}", flush=True)
            return 1
        import depth_control
        depth_control.run(args.ckpt, args.depth_lora, args.depth_image,
                          args.prompt, args.steps, args.cfg, args.mu, args.seed,
                          args.output, offload=True)
        return 0

    run(args.prompt, args.ckpt, args.steps, args.cfg, args.mu, args.width,
        args.height, args.num_images, args.seed, args.output,
        lora=args.lora, lora_scale=args.lora_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
