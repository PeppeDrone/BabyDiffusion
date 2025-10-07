# -*- coding: utf-8 -*-
"""
End-to-end: build triplets (original pose PNG, generated JPG, prompt),
cache source-pose keypoints, and fine-tune ViTPose on generated images.

Run with F5. Requires:
- generated images at ./generated_dataset/train and ./generated_dataset/val
- the original pose PNGs in TRAIN_POSE_DIR / VAL_POSE_DIR
- PROMPT_JSON (same mapping used during generation)

If YOLO fails on some samples, we fallback to full-image bbox.
"""

# --------------------------------------------------
# Imports & config
# --------------------------------------------------
import os, json, math, random, pickle
from pathlib import Path
from typing import Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation

# ------------------- USER PARAMS -------------------
TRAIN_POSE_DIR      = "./mydataset_local/train3/poses"          # original skeleton PNGs
VAL_POSE_DIR        = "./mydataset_local/validation_large"       # original skeleton PNGs (val)
PROMPT_JSON         = "./mydataset_local/train3/prompt_truncated.json"  # JSONL mapping (one json per line) {"source": "...", "prompt": "..."}
METRICS_DIR         = Path("./metrics")
GENERATED_ROOT      = Path("./generated_dataset")  # must already contain train/ and val/ with .jpg images
FT_CKPT_DIR         = Path("./fine_tuned_checkpoints")
SEED                = 44

# Model names
POSE_MODEL_ID       = "usyd-community/vitpose-base-simple"       # ViTPose
YOLO_WEIGHTS        = "yolov8s.pt"

# Training hyperparams
BATCH_SIZE          = 128
LR                  = 5e-8#5e-8
WEIGHT_DECAY        = 1e-7
EPOCHS              = 30
EARLY_STOP_PATIENCE = 3

# Resolution used to create the pose conditioning during generation (Stage-A)
IMG_RES             = 768
# --------------------------------------------------

# Determinism
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dirs
(GENERATED_ROOT / "train").mkdir(parents=True, exist_ok=True)
(GENERATED_ROOT / "val").mkdir(parents=True, exist_ok=True)
FT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# Load prompt mapping
# --------------------------------------------------
print("Loading prompt mapping...")
prompt_map = {}
with open(PROMPT_JSON, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        src = obj.get("source")
        prm = obj.get("prompt", "a professional studio photo of a person")
        if src:
            prompt_map[src] = prm

def get_prompt(fname: str) -> str:
    return prompt_map.get(fname, "a professional studio photo of a person")

# --------------------------------------------------
# Load models: YOLO (CPU ok) and ViTPose (trainable)
# --------------------------------------------------
print("Loading YOLO and ViTPose...")
yolo = YOLO(YOLO_WEIGHTS).to('cpu')  # keep on CPU
processor = AutoProcessor.from_pretrained(POSE_MODEL_ID)

# --------------------------------------------------
# Utilities to reconstruct triplets and cache source keypoints
# --------------------------------------------------
def reconstruct_pose_png_from_generated(gen_name: str) -> str:
    """
    Inverse of Stage-A naming:
      out = fname.replace('_pose','').replace('.png','.jpg')
    So here: gen 'abc.jpg' -> original pose 'abc_pose.png'
    """
    stem = Path(gen_name).stem
    return f"{stem}_pose.png"

def find_pose_path(pose_fname: str) -> Path:
    cand_train = Path(TRAIN_POSE_DIR) / pose_fname
    if cand_train.exists():
        return cand_train
    cand_val = Path(VAL_POSE_DIR) / pose_fname
    if cand_val.exists():
        return cand_val
    return None

@torch.no_grad()
def extract_keypoints_from_pose_png(pose_png_path: Path) -> np.ndarray:
    """
    Extract (K,2) keypoints from the original pose PNG using ViTPose.
    We feed a full-image bbox [0,0,W,H] (xywh) to avoid YOLO on skeletons.
    """
    img = Image.open(pose_png_path).convert("RGB")
    W, H = img.size
    full_box = np.array([[0, 0, W, H]], dtype=np.float32)  # xywh
    enc = processor(img, boxes=[full_box], return_tensors="pt").to(device)
    out = vitpose(**enc)
    post = processor.post_process_pose_estimation(out, boxes=[full_box])[0][0]
    kpts = post["keypoints"].detach().cpu().numpy()[:, :2]  # (K,2)
    return kpts

def prepare_triplets_and_cache(split: str):
    """
    For each generated JPG in GENERATED_ROOT/split, locate the original pose PNG,
    extract original-pose keypoints once, and cache them:
      <image>.srcpts.json = {
          "src_keypoints": [[x,y],...],
          "prompt": "...",
          "pose_png": "<file>",
          "img_res": IMG_RES
      }
    """
    gen_dir = GENERATED_ROOT / split
    jpgs = sorted([f for f in os.listdir(gen_dir) if f.lower().endswith(".jpg")])
    cached = 0
    prepared = 0

    print(f"Preparing triplets & caching source keypoints for [{split}]...")
    for gen_name in tqdm(jpgs):
        srcpts_path = (gen_dir / gen_name).with_suffix(".srcpts.json")
        if srcpts_path.exists():
            cached += 1
            continue

        pose_fname = reconstruct_pose_png_from_generated(gen_name)
        pose_path = find_pose_path(pose_fname)
        if pose_path is None:
            print(f"⚠️ Original pose PNG not found for {gen_name} -> expected {pose_fname}")
            continue

        try:
            src_kpts = extract_keypoints_from_pose_png(pose_path)
        except Exception as e:
            print(f"⚠️ ViTPose failed on pose PNG {pose_path}: {e}")
            continue

        payload = {
            "src_keypoints": src_kpts.tolist(),
            "prompt": get_prompt(pose_fname),
            "pose_png": pose_fname,
            "img_res": IMG_RES
        }
        with open(srcpts_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)
        prepared += 1

    print(f"[{split}] prepared: {prepared}, cached: {cached}, total: {len(jpgs)}")

# --------------------------------------------------
# Dataset: generated JPG -> inputs; labels = original pose keypoints
# --------------------------------------------------
class TripletPoseDataset(Dataset):
    """
    Each item:
      - enc: dict for ViTPose (pixel_values, boxes, etc.) built from the GENERATED JPG and YOLO bbox
      - labels: tensor (K,2) -> original-pose keypoints (rescaled to model input size)
    Notes:
      * If YOLO finds multiple people, we pick the largest bbox.
      * If YOLO fails, we fallback to full-image bbox.
    """
    def __init__(self, gen_dir: Path, processor, yolo_model, img_res: int):
        self.gen_dir = Path(gen_dir)
        self.files = sorted([f for f in os.listdir(gen_dir) if f.lower().endswith(".jpg")])
        self.processor = processor
        self.yolo = yolo_model
        self.img_res = img_res
 
    def __len__(self):
        return len(self.files)

    def _yolo_largest_person_xywh(self, pil_img: Image.Image) -> np.ndarray:
        res = self.yolo(pil_img, classes=[0], device='cpu', verbose = False)
        if len(res) == 0 or len(res[0].boxes) < 1:
            return None
        boxes = [b.xyxy[0].cpu().numpy() for b in res[0].boxes]
        areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
        x1,y1,x2,y2 = boxes[int(np.argmax(areas))]
        xywh = np.array([[x1, y1, x2-x1, y2-y1]], dtype=np.float32)
        return xywh

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = self.gen_dir / img_name
        srcpts_path = img_path.with_suffix(".srcpts.json")

        # Load generated image
        image = Image.open(img_path).convert("RGB")

        # Load original-pose keypoints (IMG_RES frame)
        with open(srcpts_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        src_kpts = np.asarray(meta["src_keypoints"], dtype=np.float32)  # (K,2)

        # Detect person bbox on generated image
        person_xywh = self._yolo_largest_person_xywh(image)
        if person_xywh is None:
            W, H = image.size
            person_xywh = np.array([[0, 0, W, H]], dtype=np.float32)

        # Build processor inputs
        enc = self.processor(image, boxes=[person_xywh], return_tensors="pt")
        # Squeeze batch dim now; the DataLoader collate will add it back
        for k in enc:
            enc[k] = enc[k].squeeze(0)

        # Labels in model input coordinate frame:
        H_in, W_in = enc["pixel_values"].shape[-2:]
        scale_x = W_in / float(self.img_res)
        scale_y = H_in / float(self.img_res)
        labels = torch.tensor(
            np.stack([src_kpts[:, 0]*scale_x, src_kpts[:, 1]*scale_y], axis=1),
            dtype=torch.float32
        )

        return enc, labels

# --------------------------------------------------
# Soft-argmax utility
# --------------------------------------------------
def softargmax_from_heatmaps(heatmaps: torch.Tensor, H_in: int, W_in: int, device) -> torch.Tensor:
    """
    heatmaps: (B, K, Hh, Wh) -> coordinates in input pixel space (H_in, W_in)
    """
    B, K, Hh, Wh = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    probs = torch.nn.functional.softmax(flat, dim=-1)

    y_idx = torch.arange(Hh, device=device).unsqueeze(1).repeat(1, Wh).contiguous().view(-1).float()
    x_idx = torch.arange(Wh, device=device).repeat(Hh).float()

    exp_y = torch.sum(probs * y_idx, dim=-1)  # (B,K)
    exp_x = torch.sum(probs * x_idx, dim=-1)  # (B,K)

    sy = H_in / float(Hh)
    sx = W_in / float(Wh)
    y = exp_y * sy
    x = exp_x * sx
    return torch.stack([x, y], dim=-1)  # (B,K,2)

def predict_and_evaluate(subject_folder: str, model, finetuned, processor, device):
    """
    Predict poses for images in a subject folder and compute error w.r.t. ground truth in COCO format.
    Saves per-frame overlay PNGs and returns [(err_base, err_fine), ...].
    """
    images_dir = os.path.join('./ext_testing_data/images/', subject_folder)
    coco_dir   = os.path.join('./ext_testing_data/coco/',   subject_folder)
    json_files = [f for f in os.listdir(coco_dir) if f.endswith(".json")]
    errors = []

    json_path = os.path.join(coco_dir, json_files[0])
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

    for annotation in coco_data["annotations"]:
        image_id   = annotation["image_id"]
        image_file = image_id_to_file[image_id]
        image_path = os.path.join(images_dir, image_file)

        frame = Image.open(image_path).convert("RGB")
        gt_keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)[:, :2]  # (K,2)

        # --- YOLO on CPU to avoid torchvision::nms CUDA issue ---
        yolo_res = yolo(frame, classes=[0], device='cpu', verbose=False)
        if len(yolo_res) == 0 or len(yolo_res[0].boxes) < 1:
            print(f"⚠️ No valid bbox for {image_file}")
            continue

        if subject_folder == 'FMS_0017_1':
            boxes  = [box.xyxy[0].cpu().numpy() for box in yolo_res[0].boxes]
            areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            xyxy   = boxes[int(np.argmax(areas))]
        else:
            xyxy   = yolo_res[0].boxes.xyxy[0].cpu().numpy()

        person_boxes = np.array([xyxy.copy()])
        person_boxes[:, 2] -= person_boxes[:, 0]
        person_boxes[:, 3] -= person_boxes[:, 1]

        enc = processor(frame, boxes=[person_boxes], return_tensors="pt").to(device)

        with torch.no_grad():
            out_base = model(**enc)
        pred_base = processor.post_process_pose_estimation(out_base, boxes=[person_boxes])[0][0]["keypoints"].cpu().numpy()

        with torch.no_grad():
            out_ft = finetuned(**enc)
        pred_ft = processor.post_process_pose_estimation(out_ft, boxes=[person_boxes])[0][0]["keypoints"].cpu().numpy()

        # overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(frame)
        plt.scatter(gt_keypoints[:,0], gt_keypoints[:,1], c='g', label='GT', s=40, marker='o')
        plt.scatter(pred_base[:,0],    pred_base[:,1],    c='r', label='Baseline', s=40, marker='x')
        plt.scatter(pred_ft[:,0],      pred_ft[:,1],      c='y', label='Fine-Tuned', s=40, marker='x')
        bbox = person_boxes[0]
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='b', facecolor='none', label='BBox')
        plt.gca().add_patch(rect)
        plt.legend()
        plt.title(f"{subject_folder} – {image_file}")
        plt.axis('off')
        out_png = METRICS_DIR / f"pose_estimation_{subject_folder}_{Path(image_file).stem}.png"
        plt.savefig(out_png, dpi=120)
        plt.close()

        # errors
        error_base = np.linalg.norm(pred_base - gt_keypoints, axis=1).mean()
        error_fine = np.linalg.norm(pred_ft   - gt_keypoints, axis=1).mean()
        errors.append((error_base, error_fine))

    return errors


# --------------------------------------------------
# Main
# --------------------------------------------------
TRAIN = 1
def main():
    if TRAIN:
        # Initialize model for training
        vitpose = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID).to(device)
        vitpose.train()

        # 1) Create caches (.srcpts.json) for train/val from original pose PNGs
        prepare_triplets_and_cache("train")
        prepare_triplets_and_cache("val")

        # 2) Datasets & loaders
        train_dataset = TripletPoseDataset(GENERATED_ROOT / "train", processor, yolo, IMG_RES)
        val_dataset   = TripletPoseDataset(GENERATED_ROOT / "val",   processor, yolo, IMG_RES)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 3) Optimizer & loss
        optimizer = torch.optim.AdamW(vitpose.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.MSELoss()

        # 4) Train loop with early stopping
        # --- before the loop ---
        train_losses, val_losses = [], []

        best_val = float("inf")
        bad = 0

        for epoch in range(1, EPOCHS+1):
            # ===== TRAIN =====
            vitpose.train()
            tr_loss = 0.0
            for enc, labels in tqdm(train_loader, desc=f"Train epoch {epoch}"):
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                out = vitpose(**enc)
                hm = out.heatmaps
                H_in, W_in = enc["pixel_values"].shape[-2:]
                preds = softargmax_from_heatmaps(hm, H_in, W_in, device)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()

            tr_loss /= max(1, len(train_loader))
            train_losses.append(tr_loss)

            # ===== VAL =====
            vitpose.eval()
            va_loss = 0.0
            with torch.no_grad():
                for enc, labels in tqdm(val_loader, desc=f"Val epoch {epoch}"):
                    enc = {k: v.to(device) for k, v in enc.items()}
                    labels = labels.to(device)
                    out = vitpose(**enc)
                    hm = out.heatmaps
                    H_in, W_in = enc["pixel_values"].shape[-2:]
                    preds = softargmax_from_heatmaps(hm, H_in, W_in, device)
                    va_loss += criterion(preds, labels).item()
            va_loss /= max(1, len(val_loader))
            val_losses.append(va_loss)

            print(f"Epoch {epoch} | train {tr_loss:.4f} | val {va_loss:.4f}")

            if va_loss < best_val:
                best_val = va_loss
                bad = 0
                # Save both formats
                torch.save(vitpose.state_dict(), FT_CKPT_DIR / "vitpose_finetuned_from_triplets.pth")
                vitpose.save_pretrained(FT_CKPT_DIR / "vitpose_finetuned_from_triplets")
                print("✓ Validation improved. Checkpoint saved.")
            else:
                bad += 1
                if bad >= EARLY_STOP_PATIENCE:
                    print("Early stopping.")
                    break
    else:
        # Load already fine-tuned model
        print("Loading fine-tuned model...")
        vitpose = VitPoseForPoseEstimation.from_pretrained(FT_CKPT_DIR / "vitpose_finetuned_from_triplets").to(device)
        vitpose.eval()

        # Load training history
        history_path = METRICS_DIR / "training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            train_losses = history.get("train_losses", [])
            val_losses = history.get("val_losses", [])
            print("Training history loaded.")
        else:
            print("Training history not found.")

    # ---- save history + plot learning curve ----
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_losses": train_losses, "val_losses": val_losses}
    with open(METRICS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "learning_curve.png", dpi=150)
    print(f"Saved learning curve to {METRICS_DIR / 'learning_curve.png'}")

    print("Done.")
    
    # ---------- Run evaluation across subjects ----------
    ext_testing_data_dir = "./ext_testing_data"
    subjects = sorted([s for s in os.listdir(os.path.join(ext_testing_data_dir, "coco")) 
                    if os.path.isdir(os.path.join(ext_testing_data_dir, "coco", s))])

    print("Evaluating baseline vs fine-tuned...")
    baseline_model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL_ID).to(device)
    baseline_model.eval()
    vitpose.eval()  # fine-tuned

    rows = []
    for subject in subjects:
        print(f"Subject: {subject}")
        errs = predict_and_evaluate(subject, baseline_model, vitpose, processor, device)
        for e_ft, e_base in errs:
            rows.append({"Subject": subject, "Error": e_base, "Model": "Baseline"})
            rows.append({"Subject": subject, "Error": e_ft,   "Model": "Fine-Tuned"})

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "pose_estimation_errors.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-frame errors to {csv_path}")

    # optional Excel (requires openpyxl)
    try:
        xlsx_path = METRICS_DIR / "pose_estimation_errors.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"Saved per-frame errors to {xlsx_path}")
    except Exception as e:
        print(f"(Excel save skipped: {e})")

    # ----------- Boxplot -----------
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Subject", y="Error", hue="Model", showfliers=False)
    plt.ylabel("Pixel Error (avg L2 over joints)")
    plt.xlabel("Subject")
    plt.title("Baseline vs Fine-Tuned – Pose Error by Subject")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    boxplot_path = METRICS_DIR / "error_comparison_boxplot_by_subject.svg"
    plt.savefig(boxplot_path)
    print(f"Saved boxplot to {boxplot_path}")

    # Save the data used for the plot
    boxplot_data_path = METRICS_DIR / "error_comparison_boxplot_data.csv"
    df.to_csv(boxplot_data_path, index=False)
    print(f"Saved boxplot data to {boxplot_data_path}")

    # ----------- Paired t-tests -----------
    print("\nPaired t-tests (Baseline vs Fine-Tuned):")
    for subject in subjects:
        b = df[(df.Subject==subject) & (df.Model=="Baseline")]["Error"].values
        f = df[(df.Subject==subject) & (df.Model=="Fine-Tuned")]["Error"].values
        n = min(len(b), len(f))
        if n >= 2:
            t, p = ttest_rel(b[:n], f[:n])
            print(f"  {subject:>20s}: t = {t:.4f}, p = {p:.3e} (n={n})")
        else:
            print(f"  {subject:>20s}: not enough paired frames (n={n})")

    # Overall paired test (pool frames, then pair by index within subject)
    pairs_base, pairs_ft = [], []
    for subject in subjects:
        b = df[(df.Subject==subject) & (df.Model=="Baseline")]["Error"].values
        f = df[(df.Subject==subject) & (df.Model=="Fine-Tuned")]["Error"].values
        n = min(len(b), len(f))
        pairs_base.extend(b[:n])
        pairs_ft.extend(f[:n])
    if len(pairs_base) >= 2:
        t, p = ttest_rel(pairs_base, pairs_ft)
        print(f"\nOverall (all paired frames): t = {t:.4f}, p = {p:.3e}, N = {len(pairs_base)}")
    else:
        print("\nOverall: not enough paired frames.")

    


    
        
if __name__ == "__main__":
    main()
