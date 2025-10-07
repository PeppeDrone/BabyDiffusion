"""
4x2 grid di predizioni di validazione (2 per modello) per le cartelle in D:/vit.
All fp16 (come nel codice che funziona).
"""

import os
import re
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# -----------------------------
# CONFIGURAZIONE
# -----------------------------
BASE_DIR = "D:/vit"  # cartelle come nello screenshot
SD_BACKBONE = "stabilityai/stable-diffusion-2-1"
LOCAL_FILES_ONLY = True

VAL_POSE_DIR   = "./mydataset_local/image_test"      # *_pose.png
TRAIN_PROMPTS  = "./mydataset_local/train3/prompt_truncated.json"  # JSONL: {"source": "<file>", "prompt": "<text>"}

# Due ID (basename senza _pose.png); se vuoto usa i primi due trovati
SELECTED_VAL_IMAGE_IDS = []

NUM_INFERENCE_STEPS = 250
GUIDANCE_SCALE      = 6.0
CONTROL_SCALE       = 0.6
SEED = 10

# -----------------------------
# UTILITIES
# -----------------------------
def set_seed(seed: int):
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)
    return g

def list_eight_models(base_dir: str):
    cand = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("controlnet"):
            cand.append(full)
    cand.sort()
    return cand[:8]

def parse_params_from_name(name: str):
    bs = ga = res = None
    lr = None
    m_bs = re.search(r"_bs(\d+)", name)
    m_ga = re.search(r"_ga(\d+)", name)
    m_res = re.search(r"_res(\d+)", name)
    m_lr = re.search(r"_lr([0-9eE\.\-]+)", name)
    if m_bs: bs = int(m_bs.group(1))
    if m_ga: ga = int(m_ga.group(1))
    if not m_res:
        raise ValueError(f"Nessun 'res' nel nome cartella: {name}")
    res = int(m_res.group(1))
    if m_lr: lr = m_lr.group(1)
    return dict(bs=bs, ga=ga, res=res, lr=lr)

def load_validation():
    with open(TRAIN_PROMPTS, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]

    data = []
    for fname in os.listdir(VAL_POSE_DIR):
        if not fname.endswith("_pose.png"):
            continue
        base = fname[:-9]  # rimuove '_pose.png'
        p = next((x for x in prompts if x["source"] == fname), None)
        if p:
            data.append({
                "image_id": base,
                "control_image": os.path.join(VAL_POSE_DIR, fname),
                "prompt": p["prompt"]
            })
    data.sort(key=lambda d: d["image_id"])
    return data

def pick_two(validation, ids):
    if ids:
        out = []
        for sid in ids:
            hit = next((d for d in validation if d["image_id"] == sid), None)
            if hit: out.append(hit)
        if len(out) >= 2:
            print('Using selected images')
            return out[:2]
    return validation[:2]

def assert_nonzero_weights(model, name="controlnet"):
    # quick sanity check: parameters should have non-trivial std
    with torch.no_grad():
        t = None
        for p in model.parameters():
            if p is not None and p.numel() > 0:
                t = p.detach()
                break
        if t is None:
            raise RuntimeError(f"{name}: no parameters found")
        std = t.float().std().item()
        if not np.isfinite(std) or std < 1e-6:
            raise RuntimeError(f"{name}: suspicious weights (std={std:.3e}) – likely uninitialized")

def load_pipe(model_dir):
    """
    Load *actual* weights on CUDA, all fp16 (matches your working code).
    Avoid device='meta' + to_empty to prevent uninitialized weights.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ControlNet with real weights
    controlnet = ControlNetModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        local_files_only=LOCAL_FILES_ONLY
    ).to(device)

    # Sanity check: ensure weights are not all zeros
    assert_nonzero_weights(controlnet, name=f"controlnet@{os.path.basename(model_dir)}")

    # Build pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_BACKBONE,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        local_files_only=LOCAL_FILES_ONLY
    ).to(device)

    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    return pipe


def infer_one(pipe, prompt, control_path, res, guidance, cond_scale, steps, generator):
    control = Image.open(control_path).convert("RGB").resize((res, res))
    with torch.inference_mode():
        out = pipe(
            prompt,
            control,                       # come nel tuo codice funzionante
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=cond_scale,
            height=res,                    # esplicito
            width=res,
            generator=generator
        )
    return out.images[0]

def concat_h(a: Image.Image, b: Image.Image):
    h = max(a.height, b.height)
    w = a.width + b.width
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    return canvas

# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs("./figures", exist_ok=True)

    model_dirs = list_eight_models(BASE_DIR)
    if len(model_dirs) == 0:
        raise RuntimeError(f"Nessun modello trovato in {BASE_DIR}")
    if len(model_dirs) < 8:
        print(f"⚠️ Trovati solo {len(model_dirs)} modelli; i pannelli restanti resteranno vuoti.")

    val = load_validation()
    if len(val) < 2:
        raise RuntimeError("Servono almeno due item di validazione.")
    chosen = pick_two(val, SELECTED_VAL_IMAGE_IDS)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "figure.dpi": 120})

    base_gen = set_seed(SEED)

    for i in range(8):
        ax = axes[i]
        ax.axis("off")

        if i >= len(model_dirs):
            ax.set_title("—")
            continue

        mdir = model_dirs[i]
        fname = os.path.basename(mdir)
        try:
            params = parse_params_from_name(fname)
        except Exception as e:
            ax.text(0.5, 0.5, f"Name parse error:\n{e}", ha="center", va="center", wrap=True)
            continue

        res = 512#params["res"]

        try:
            pipe = load_pipe(mdir)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", ha="center", va="center", wrap=True)
            continue

        try:
            img1 = infer_one(pipe, chosen[0]["prompt"], chosen[0]["control_image"],
                             res, GUIDANCE_SCALE, CONTROL_SCALE, NUM_INFERENCE_STEPS, base_gen)
            # Save individual images
            img1.save(f"./figures/{fname}_sample1_{chosen[0]['image_id']}.png")
            # avanza il generatore per il secondo sample
            _ = torch.rand(1, generator=base_gen,
                           device=base_gen.device if hasattr(base_gen, "device") else None)

            img2 = infer_one(pipe, chosen[1]["prompt"], chosen[1]["control_image"],
                             res, GUIDANCE_SCALE, CONTROL_SCALE, NUM_INFERENCE_STEPS, base_gen)
            img2.save(f"./figures/{fname}_sample2_{chosen[1]['image_id']}.png")

            tile = concat_h(img1, img2)
            ax.imshow(tile)

            title = f"BS={params['bs']} | GA={params['ga']} | RES={res} | LR={params['lr']}"
            ax.set_title(title)

        except Exception as e:
            ax.text(0.5, 0.5, f"Inference error:\n{e}", ha="center", va="center", wrap=True)

        finally:
            # cleanup VRAM
            try:
                del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    fig.suptitle(
        f"Validation predictions (2 per modello) | steps={NUM_INFERENCE_STEPS}, guidance={GUIDANCE_SCALE}, cond={CONTROL_SCALE}",
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = "./figures/val_grid_4x2.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ Salvato: {out_path}")

if __name__ == "__main__":
    main()
