"""
Krea 2 Turbo Studio - a GGF (Get Going Fast) standalone GUI for Krea-2.

Runs as a small Gradio front-end that shells out to gen.py (a memory-managed
runner) so it works on a 24 GB GPU. Each generation runs as a fresh process,
so no model is held in VRAM between runs.

Launched by run.bat using the venv python inside krea-2\\.venv.
"""

import os
import sys
import glob
import datetime
import subprocess

import gradio as gr

# --------------------------------------------------------------------------
# Paths / config  (this file lives OUTSIDE the cloned repo, next to run.bat)
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE  # workflow.py now lives inside the krea-2 repo
MODELS_DIR = os.path.join(REPO, "models")
OUTPUTS_DIR = os.path.join(REPO, "outputs")
LOGS_DIR = os.path.join(REPO, "logs")
INFERENCE = os.path.join(REPO, "inference.py")
GEN = os.path.join(HERE, "gen.py")  # memory-managed runner (fits a 24 GB GPU)

# Selectable checkpoints: label -> (local filename, download url).
MODELS = {
    "Turbo FP8  -  ~12 GB, fits 24 GB, faster (recommended)": (
        "krea2_turbo_fp8.safetensors",
        "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/"
        "krea2_turbo_fp8.safetensors?download=true",
    ),
    "Turbo bf16  -  ~26 GB, full precision, streams from RAM": (
        "turbo.safetensors",
        "https://huggingface.co/krea/Krea-2-Turbo/resolve/main/"
        "turbo.safetensors?download=true",
    ),
}

# Depth ControlNet-LoRA (image-guided). Needs the bf16 base + depth_control.py.
BF16_MODEL = "turbo.safetensors"
DEPTH_LORA_FILE = "depth-control-lora.safetensors"
DEPTH_LORA_URL = ("https://huggingface.co/Patil/Krea-2-depth-controlnet/"
                  "resolve/main/depth-control-lora.safetensors")

APP_TITLE = "Krea 2 Turbo Studio"
APP_SUBTITLE = (
    "AI success starts with the right setup. This standalone Krea-2 Turbo "
    "workspace is packaged for Windows/NVIDIA users who want fast installs, "
    "tested model loading, and a clean text-to-image workflow - pick the FP8 "
    "checkpoint for a 24 GB card, add a LoRA, and prompt."
)

_current_proc = None  # tracks a running generation subprocess


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def model_path(label):
    return os.path.join(MODELS_DIR, MODELS[label][0])


def model_exists(label):
    return os.path.isfile(model_path(label))


def default_model_label():
    """Prefer FP8 if present, else any present model, else FP8 (to download)."""
    present = [lbl for lbl in MODELS if model_exists(lbl)]
    for lbl in MODELS:  # MODELS is ordered FP8 first
        if lbl in present:
            return lbl
    return next(iter(MODELS))


def initial_status():
    if not os.path.isfile(INFERENCE):
        return ("Krea-2 is not installed yet.\n"
                "Run install.bat in this folder first, then restart.")
    have = [MODELS[l][0] for l in MODELS if model_exists(l)]
    if not have:
        return ("No model found in models\\.\n"
                "Pick a model below and click 'Download / Repair Model'.")
    return "Models present: " + ", ".join(have)


def find_aria2():
    local = os.path.join(HERE, "aria2c.exe")
    return local if os.path.exists(local) else None


def _clean_path(p):
    return (p or "").strip().strip('"').strip("'")


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
def download_model(label):
    """Download / repair the selected checkpoint with aria2c (or curl).

    Progress streams to the console window that launched run.bat. Click again
    to resume/repair - aria2c continues a partial file.
    """
    filename, url = MODELS[label]
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, filename)
    aria2 = find_aria2()
    if aria2:
        cmd = [aria2, "-x16", "-s16", "-k1M", "--continue=true",
               "--auto-file-renaming=false", "--allow-overwrite=true",
               "--disable-ipv6=true", "-d", MODELS_DIR, "-o", filename, url]
    else:
        cmd = ["curl", "-L", "-o", dest, url]
    try:
        rc = subprocess.call(cmd, cwd=HERE)
    except Exception as e:
        return f"Download failed to start: {e}"
    if rc == 0 and os.path.isfile(dest):
        return f"Model ready at {dest}"
    return ("Download / repair exited with code %d. Check the server console, "
            "then click Download / Repair Model again to resume." % rc)


# tiny script run in a separate process so tkinter never touches Gradio's
# worker threads (which can crash the dialog on Windows).
_BROWSE_SCRIPT = (
    "import tkinter as tk, tkinter.filedialog as fd\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "p = fd.askopenfilename(title='Select a LoRA .safetensors', "
    "filetypes=[('Safetensors', '*.safetensors'), ('All files', '*.*')])\n"
    "r.destroy(); print(p or '')\n"
)


def browse_lora(current):
    """Open a native Windows file picker (filtered to .safetensors) and return
    the chosen path. Keeps the current value if the dialog is cancelled."""
    try:
        out = subprocess.run([sys.executable, "-c", _BROWSE_SCRIPT],
                             capture_output=True, text=True, timeout=600)
        path = (out.stdout or "").strip()
        return path if path else current
    except Exception:
        return current


def download_depth_lora():
    """Fetch the depth ControlNet-LoRA into models\\. Returns (path, status)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, DEPTH_LORA_FILE)
    aria2 = find_aria2()
    if aria2:
        cmd = [aria2, "-x16", "-s16", "-k1M", "--continue=true",
               "--auto-file-renaming=false", "--allow-overwrite=true",
               "--disable-ipv6=true", "-d", MODELS_DIR, "-o", DEPTH_LORA_FILE,
               DEPTH_LORA_URL]
    else:
        cmd = ["curl", "-L", "-o", dest, DEPTH_LORA_URL]
    try:
        rc = subprocess.call(cmd, cwd=HERE)
    except Exception as e:
        return "", f"Depth LoRA download failed to start: {e}"
    if rc == 0 and os.path.isfile(dest):
        return dest, f"Depth LoRA ready at {dest}"
    return "", f"Depth LoRA download exited with code {rc} (click again to resume)."


def open_output_folder():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    try:
        os.startfile(OUTPUTS_DIR)  # Windows only
        return f"Opened {OUTPUTS_DIR}"
    except Exception as e:
        return f"Could not open folder: {e}"


def stop_generation():
    global _current_proc
    if _current_proc is not None and _current_proc.poll() is None:
        try:
            _current_proc.terminate()
            return "Stopped the running generation."
        except Exception as e:
            return f"Could not stop the process: {e}"
    return ("Nothing is running. Each image runs as a fresh process, so no "
            "model is held in VRAM between runs.")


def generate(prompt, model_label, lora, lora_scale, steps, cfg, mu,
             width, height, num_images, seed, depth_image, depth_lora):
    global _current_proc
    if not os.path.isfile(INFERENCE):
        return None, "Krea-2 is not installed. Run install.bat first."
    if not prompt or not prompt.strip():
        return None, "Enter a prompt first."

    depth_image = (depth_image or "").strip() if isinstance(depth_image, str) else None
    depth_lora = _clean_path(depth_lora)
    depth_mode = bool(depth_image and depth_lora)

    if depth_mode:
        # Depth ControlNet-LoRA is image-guided; works with the selected fp8/bf16 base.
        ckpt = model_path(model_label)
        if not os.path.isfile(ckpt):
            return None, (f"{os.path.basename(ckpt)} not found. Select it and click "
                          "'Download / Repair Model' first.")
        if not os.path.isfile(depth_lora):
            return None, f"Depth LoRA not found: {depth_lora}"
    else:
        ckpt = model_path(model_label)
        if not os.path.isfile(ckpt):
            return None, (f"{os.path.basename(ckpt)} not found. Select it and click "
                          "'Download / Repair Model' first.")
        lora = _clean_path(lora)
        if lora and not os.path.isfile(lora):
            return None, f"LoRA file not found: {lora}"

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_prefix = os.path.join(OUTPUTS_DIR, f"krea_{stamp}")

    cmd = [
        sys.executable, GEN, prompt.strip(),
        "--ckpt", ckpt,
        "--steps", str(int(steps)),
        "--cfg", str(float(cfg)),
        "--mu", str(float(mu)),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--num-images", str(int(num_images)),
        "--seed", str(int(seed)),
        "--output", out_prefix,
    ]
    if depth_mode:
        cmd += ["--depth-image", depth_image, "--depth-lora", depth_lora]
    elif lora:
        cmd += ["--lora", lora, "--lora-scale", str(float(lora_scale))]

    # gen.py auto-detects fp8 vs bf16 and picks the VRAM strategy. The Qwen
    # text encoder and VAE are pulled from Hugging Face on first use and cached.
    env = dict(
        os.environ,
        PYTHONPATH=REPO,
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
    )
    log_path = os.path.join(LOGS_DIR, f"gen-{stamp}.log")
    tail = []
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            _current_proc = subprocess.Popen(
                cmd, cwd=REPO, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            for line in _current_proc.stdout:
                lf.write(line)
                tail.append(line)
                if len(tail) > 60:
                    tail.pop(0)
            _current_proc.wait()
        rc = _current_proc.returncode
    except Exception as e:
        _current_proc = None
        return None, f"Failed to launch generation: {e}"
    finally:
        _current_proc = None

    images = sorted(glob.glob(out_prefix + "_*.png"))
    if rc == 0 and images:
        note = f"Done. Saved {len(images)} image(s) to {OUTPUTS_DIR}"
        if lora:
            note += f"  (LoRA applied @ {lora_scale})"
        return images[0], note
    return None, ("Generation failed (exit %s).\nLast log lines:\n%s"
                  % (rc, "".join(tail)[-1800:]))


# --------------------------------------------------------------------------
# Style  (matches the GGF "Studio" look: cream page, teal gradient header)
# --------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important;
  background: #f5efe1 !important; }
footer { display: none !important; }

#ggf-header { border-radius: 18px; padding: 30px 34px; margin: 6px 0 18px 0;
  color: #fff; background: linear-gradient(118deg, #2a5d70 0%, #234653 48%, #18222b 100%);
  box-shadow: 0 8px 26px rgba(20,40,50,.18); }
.ggf-brandrow { display: flex; align-items: center; gap: 12px; }
.ggf-logo { width: 40px; height: 40px; border-radius: 11px; display: flex;
  align-items: center; justify-content: center; font-weight: 800; font-size: 13px;
  color: #fff; letter-spacing: .5px;
  background: linear-gradient(135deg, #e0593a, #d23f28);
  box-shadow: 0 3px 8px rgba(210,63,40,.4); }
.ggf-brandtext { font-weight: 800; letter-spacing: 2px; font-size: 14px; color: #e7a657; }
.ggf-title { font-size: 38px; font-weight: 800; margin: 16px 0 6px 0; color: #fff; }
.ggf-subtitle { max-width: 880px; line-height: 1.55; color: #cdd8dd; font-size: 15px; }
.ggf-links { margin: 16px 0 4px 0; }
.ggf-links a { color: #e7a657; font-weight: 700; text-decoration: underline;
  margin-right: 20px; }
.ggf-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.ggf-pill { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18);
  color: #e7eef1; padding: 7px 15px; border-radius: 999px; font-size: 13px; }

/* action button row */
.ggf-actbtn button { background: #fffdf8 !important; color: #34302a !important;
  border: 1px solid #e6dcc6 !important; border-radius: 12px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.04) !important; font-weight: 600 !important;
  min-height: 64px !important; }
.ggf-actbtn button:hover { background: #fbf4e6 !important; border-color: #ddca9f !important; }

/* primary generate */
.ggf-generate button { background: linear-gradient(135deg, #e0593a, #d23f28) !important;
  color: #fff !important; border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; min-height: 50px !important; font-size: 16px !important; }
.ggf-generate button:hover { filter: brightness(1.05); }

/* panels: status, accordions, group cards */
.ggf-card, .gr-accordion, .gr-group { background: #fffdf9 !important;
  border: 1px solid #e6dcc6 !important; border-radius: 14px !important; }
#ggf-status textarea { background: #fffdf9 !important; border-radius: 10px !important;
  color: #34302a !important; }
"""

HEADER_HTML = f"""
<div id="ggf-header">
  <div class="ggf-brandrow">
    <div class="ggf-logo">GGF</div>
    <div class="ggf-brandtext">GET GOING FAST</div>
  </div>
  <div class="ggf-title">{APP_TITLE}</div>
  <div class="ggf-subtitle">{APP_SUBTITLE}</div>
  <div class="ggf-links">
    <a href="https://getgoingfast.pro" target="_blank" rel="noopener">GetGoingFast.pro</a>
    <a href="https://www.youtube.com/@cognibuild" target="_blank" rel="noopener">Cognibuild on YouTube</a>
  </div>
  <div class="ggf-pills">
    <span class="ggf-pill">Turbo 8-step</span>
    <span class="ggf-pill">FP8 or bf16</span>
    <span class="ggf-pill">LoRA support</span>
    <span class="ggf-pill">Windows + NVIDIA focused</span>
  </div>
</div>
"""

EXAMPLE_PROMPT = (
    "a fox walking in the snow at golden hour, cinematic photo, soft "
    "volumetric light, shallow depth of field, highly detailed, "
    "masterpiece quality"
)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
theme = gr.themes.Soft(primary_hue="orange", secondary_hue="orange",
                       neutral_hue="stone")

with gr.Blocks(title=APP_TITLE) as demo:
    gr.HTML(HEADER_HTML)

    # action row + status
    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            with gr.Row():
                btn_download = gr.Button("Download / Repair Model",
                                         elem_classes="ggf-actbtn")
                btn_stop = gr.Button("Stop Generation", elem_classes="ggf-actbtn")
                btn_open = gr.Button("Open Output Folder",
                                     elem_classes="ggf-actbtn")
        with gr.Column(scale=2):
            status = gr.Textbox(label="Status", value=initial_status(),
                                lines=4, interactive=False, elem_id="ggf-status")

    with gr.Accordion("Model", open=True):
        model_dd = gr.Dropdown(
            choices=list(MODELS.keys()), value=default_model_label(),
            label="Checkpoint",
            info="FP8 fits a 24 GB GPU and is faster. bf16 is full precision "
                 "and streams from system RAM. 'Download / Repair Model' fetches "
                 "the one selected here.")

    # main: prompt | output
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=10, value=EXAMPLE_PROMPT)
            btn_generate = gr.Button("Generate", elem_classes="ggf-generate")
        with gr.Column(scale=1):
            output = gr.Image(label="Output", height=520)

    with gr.Accordion("LoRA (optional)", open=False):
        with gr.Row():
            lora_path = gr.Textbox(
                label="LoRA file (.safetensors)", scale=5,
                placeholder=r"Click Browse... or paste a path  -  leave blank for none")
            btn_browse = gr.Button("Browse...", scale=1)
        lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                               label="LoRA strength")
        gr.Markdown("Krea trains LoRAs on Raw and applies them on Turbo. The "
                    "LoRA is merged into the checkpoint weights at load time; "
                    "common key formats (lora_A/B, lora_down/up, kohya) are "
                    "auto-matched. The console prints how many modules matched.")

    with gr.Accordion("Depth control (ControlNet-LoRA, optional)", open=False):
        gr.Markdown("Image-guided: the DEPTH of your input image steers the "
                    "output. Works with your selected model (fp8 or bf16) + the "
                    "depth LoRA, and on first use downloads Depth-Anything-V2 "
                    "(~1.3 GB). Upload an image AND set the depth LoRA to enable "
                    "it; leave the image empty for normal text-to-image. Turbo "
                    "defaults (8 steps, CFG 0, mu 1.15) work here too.")
        depth_image = gr.Image(label="Input image (depth guidance)",
                               type="filepath", sources=["upload"], height=200)
        with gr.Row():
            depth_lora_path = gr.Textbox(
                label="Depth LoRA (.safetensors)", scale=4,
                placeholder="Click Download depth LoRA, or paste a path")
            btn_dl_depth = gr.Button("Download depth LoRA", scale=1)

    with gr.Accordion("Generation settings (Turbo)", open=False):
        with gr.Row():
            steps = gr.Slider(1, 16, value=8, step=1, label="Steps")
            cfg = gr.Slider(0.0, 7.0, value=0.0, step=0.5,
                            label="Guidance (CFG)  -  Turbo uses 0")
            mu = gr.Slider(0.5, 2.0, value=1.15, step=0.05,
                           label="Timestep shift (mu)")
        with gr.Row():
            width = gr.Slider(1024, 2048, value=1024, step=64, label="Width")
            height = gr.Slider(1024, 2048, value=1024, step=64, label="Height")
        with gr.Row():
            num_images = gr.Slider(1, 4, value=1, step=1, label="Number of images")
            seed = gr.Number(label="Seed", value=0, precision=0)
        gr.Markdown("Turbo defaults: 8 steps, CFG 0, mu 1.15. First generation "
                    "downloads the Qwen text encoder (~8 GB) and the VAE - a "
                    "one-time cache step. Keep resolution near 1024 on a 24 GB "
                    "card.")

    # wiring
    btn_generate.click(
        generate,
        inputs=[prompt, model_dd, lora_path, lora_scale, steps, cfg, mu,
                width, height, num_images, seed, depth_image, depth_lora_path],
        outputs=[output, status],
    )
    btn_download.click(download_model, inputs=[model_dd], outputs=[status])
    btn_browse.click(browse_lora, inputs=[lora_path], outputs=[lora_path])
    btn_dl_depth.click(download_depth_lora, outputs=[depth_lora_path, status])
    btn_stop.click(stop_generation, outputs=[status])
    btn_open.click(open_output_folder, outputs=[status])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                theme=theme, css=CSS)
