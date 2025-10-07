import os
import json
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import ImageDraw
import pickle
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
CHECKPOINTS_DIR = "D:/checks"  # Your ControlNet checkpoints
VALIDATION_DIR = './mydataset_local/validation_large'
TRAIN_IMAGES_DIR = './mydataset_local/train3/images'
PROMPT_JSON_PATH = './mydataset_local/train3/prompt_truncated.json'
OUTPUT_ROOT = './visual_comparison'
OUTPUT_SUBDIR = 'downstream'
RESOLUTION = 512
GUIDANCE_SCALE = 6
STEPS = 200
KEYPOINT_SCORE_THRESH = 0.6
skeleton_connections = [
        (15, 13), (13, 11),        # left leg: ankle->knee, knee->hip
        (16, 14), (14, 12),        # right leg: ankle->knee, knee->hip
        (11, 12),                  # pelvis: left hip->right hip
        (5, 11), (6, 12), (5, 6),  # torso: left shoulder->left hip, right shoulder->right hip, shoulder->shoulder
        (5, 7), (7, 9),            # left arm: shoulder->elbow, elbow->wrist
        (6, 8), (8, 10),           # right arm: shoulder->elbow, elbow->wrist
        (1, 2), (0, 1), (0, 2),    # head: eyes->each other and nose->eyes
        (1, 3), (2, 4),            # face: left eye->left ear, right eye->right ear
        (3, 5), (4, 6)             # neck: left ear->left shoulder, right ear->right shoulder
    ]
    
# Define colors for different body parts (BGR format for OpenCV)
color_map = {
    'left_leg':   (0, 255, 0),    # green
    'right_leg':  (255, 0, 0),    # blue
    'torso':      (255, 255, 255),# white
    'left_arm':   (0, 255, 255),  # yellow
    'right_arm':  (255, 0, 255),  # magenta
    'face':       (0, 0, 255)     # red
}

# Assign body part groups to connections
connection_groups = [
    'left_leg', 'left_leg',
    'right_leg','right_leg',
    'torso',
    'torso','torso','torso',
    'left_arm','left_arm',
    'right_arm','right_arm',
    'face','face','face',
    'face','face',
    'torso','torso'
]

# Accuracy metric
def mpjpe(pred, gt):
    return np.mean(np.linalg.norm(pred - gt, axis=1))

def pck(pred, gt, threshold=20):
    return np.mean(np.linalg.norm(pred - gt, axis=1) < threshold)

# # ========== PREPARE OUTPUT ==========
output_path = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIR)
os.makedirs(output_path, exist_ok=True)

# ========== LOAD PROMPTS ==========
with open(PROMPT_JSON_PATH, 'r') as f:
    all_prompts = [json.loads(line) for line in f]
prompt_map = {entry['source']: entry['prompt'] for entry in all_prompts}

# ========== VALIDATION ENTRIES ==========
validation_entries = []
for fname in sorted(os.listdir(VALIDATION_DIR)):
    if fname.endswith('_pose.png') and fname in prompt_map:
        validation_entries.append({
            'image_id': fname.replace('_pose.png', ''),
            'control_image_path': os.path.join(VALIDATION_DIR, fname),
            'prompt': prompt_map[fname]
        })
print(f"Found {len(validation_entries)} validation samples.")

# ========== LOAD CONTROLNET PIPELINE ==========
print("🚀 Loading ControlNet model...")
controlnet = ControlNetModel.from_pretrained(
    os.path.join(CHECKPOINTS_DIR, 'newborn'),
    torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to('cuda')

# ========== LOAD VIT POSE + YOLO ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

pose_model_name = "usyd-community/vitpose-base-simple"
print(f"Loading ViT Pose model: {pose_model_name}...")
processor = AutoProcessor.from_pretrained(pose_model_name)
pose_model = VitPoseForPoseEstimation.from_pretrained(pose_model_name).to(device)
pose_model.eval()

print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8s.pt").to('cpu')

# ========== POSE EXTRACTION FUNCTION ==========
def extract_pose_keypoints(image_pil):
    """Return keypoints (N,2) or None if detection fails."""
    results = yolo_model(image_pil, classes=[0])
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    xyxy = results[0].boxes.xyxy[0].cpu().numpy()
    person_boxes = np.array([xyxy])
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    inputs = processor(image_pil, boxes=[person_boxes], return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    dataset_index = torch.tensor([5], device=device)
    with torch.no_grad():
        outputs = pose_model(**inputs, dataset_index=dataset_index)
    pose_results = processor.post_process_pose_estimation(outputs, boxes=[person_boxes])

    pose = pose_results[0][0]
    keypoints = pose['keypoints'].cpu().numpy()
    scores = pose['scores'].cpu().numpy()
    # Ensure all scores are > 0.5
    if np.any(scores < KEYPOINT_SCORE_THRESH):
        return None
    valid_idx = scores >= KEYPOINT_SCORE_THRESH
    if np.sum(valid_idx) < 5:
        return None
    return keypoints

# ========== GENERATION & EVALUATION LOOP ==========
mpjpe_scores, pck_scores = [], []

print("🔄 Generating images and evaluating poses...")
for idx, entry in enumerate(tqdm(validation_entries)):
    try:
        image_id = entry['image_id']
        prompt = entry['prompt']
        control_img = Image.open(entry['control_image_path']).resize((RESOLUTION, RESOLUTION))

        # Load GT image corresponding to control image but without '_pose'
        gt_image_name = entry['control_image_path'].split('/')[-1].replace('_pose.png', '.jpg').split("\\")[1]
        gt_image_path = os.path.join(TRAIN_IMAGES_DIR, gt_image_name)
        gt_image = Image.open(gt_image_path).resize((RESOLUTION, RESOLUTION))

        # Generate new image
        img_newborn = pipe(
            prompt,
            image=control_img,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images[0]

        # Extract poses (GT from corresponding original image, and generated)
        gt_kpts = extract_pose_keypoints(gt_image)
        gen_kpts = extract_pose_keypoints(img_newborn)
        if gt_kpts is None:
            continue
        if gen_kpts is None:
            continue

        if gt_kpts is not None and gen_kpts is not None:
            min_joints = min(gt_kpts.shape[0], gen_kpts.shape[0])
            mpjpe_scores.append(mpjpe(gen_kpts[:min_joints], gt_kpts[:min_joints]))
            pck_scores.append(pck(gen_kpts[:min_joints], gt_kpts[:min_joints]))
            print(mpjpe_scores[-1],pck_scores[-1])
        # Visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(gt_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(control_img)
        axes[1].set_title('Original Pose')
        axes[1].axis('off')

        axes[2].imshow(img_newborn)
        axes[2].set_title('Generated Image')
        axes[2].axis('off')



        gen_img_with_pose = np.array(img_newborn.copy())
        skeleton_img = np.zeros_like(gen_img_with_pose)        
        for (idx, (i, j)) in enumerate(skeleton_connections):
                pt1 = tuple(int(v) for v in gen_kpts[i])
                pt2 = tuple(int(v) for v in gen_kpts[j])
                group = connection_groups[idx]
                color = color_map[group]
                cv2.line(skeleton_img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
            
        for k, (x, y) in enumerate(gen_kpts):
                cv2.circle(skeleton_img, (int(x), int(y)), radius=3, color=(0,0,255), thickness=-1)
                

        axes[3].imshow(skeleton_img)
        axes[3].set_title('Generated Image Pose')
        axes[3].axis('off')

        # Save visualization
        visualization_path = os.path.join(output_path, f"{image_id}_comparison.png")
        plt.savefig(visualization_path)
        plt.close()

        # Save generated image
        img_newborn.save(os.path.join(output_path, f"{image_id}_gen.png"))
        
        

    except Exception as e:
        print(f"❌ Error processing {entry['image_id']}: {e}")

print(f"\n✅ Evaluation complete!")
print(f"Mean MPJPE: {np.mean(mpjpe_scores):.2f} px")
print(f"Mean PCK: {np.mean(pck_scores):.2%}")
# Save metrics as pickle
metrics_path = os.path.join(output_path, "metrics.pkl")
with open(metrics_path, "wb") as f:
    pickle.dump({"mpjpe_scores": mpjpe_scores, "pck_scores": pck_scores}, f)

# Load metrics from pickle file
metrics_path = os.path.join(output_path, "metrics.pkl")
with open(metrics_path, "rb") as f:
    metrics = pickle.load(f)

mpjpe_scores = metrics["mpjpe_scores"]
pck_scores = [m*100 for m in metrics["pck_scores"]]

# Create boxplot for metrics
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([mpjpe_scores, pck_scores], labels=["MPJPE", "PCK"])
ax.set_title("Metrics Boxplot")
ax.set_ylabel("Scores")
plt.savefig(os.path.join(output_path, "metrics_boxplot.png"))
plt.show()
plt.close()
# Create boxplot for metrics with dual y-axis
fig, axes = plt.subplots(1, 2, figsize=(7,4))

# Plot MPJPE on the first panel
axes[0].boxplot([mpjpe_scores], labels=["MPJPE"], showfliers=False)
axes[0].set_ylabel("MPJPE, No. of pixels")
axes[0].tick_params(axis="x", width=2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['left'].set_linewidth(2)
axes[0].spines['bottom'].set_linewidth(2)
axes[0].set_xlabel("")

# Plot PCK on the second panel
axes[1].boxplot([pck_scores], labels=["PCK"], showfliers=False)
axes[1].set_ylabel("PCK, %")
axes[1].tick_params(axis="x", width=2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_linewidth(2)
axes[1].spines['bottom'].set_linewidth(2)
axes[1].set_xlabel("")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_path, "metrics_boxplot_separate_panels.png"))
plt.show()
plt.close()
# Calculate and print median and IQR for MPJPE
mpjpe_median = np.median(mpjpe_scores)
mpjpe_iqr = np.percentile(mpjpe_scores, 75) - np.percentile(mpjpe_scores, 25)
print(f"MPJPE Median: {mpjpe_median:.2f}")
print(f"MPJPE IQR: {mpjpe_iqr:.2f}")

# Calculate and print median and IQR for PCK
pck_median = np.median(pck_scores)
pck_iqr = np.percentile(pck_scores, 75) - np.percentile(pck_scores, 25)
print(f"PCK Median: {pck_median:.2f}%")
print(f"PCK IQR: {pck_iqr:.2f}%")


# Define pixel thresholds
pixel_thresholds = [2, 5, 10, 15, 20]

# Calculate PCK at each threshold
pck_values = []
pck_stds = []
for threshold in pixel_thresholds:
    pck_at_threshold = [np.array(mpjpe_scores) < threshold]
    per_points = 100*np.argwhere(pck_at_threshold).shape[0]/len(mpjpe_scores)
    
    pck_values.append(per_points)  # Convert to percentage

# Plot PCK curve with error bars
plt.figure(figsize=(6,4))
plt.scatter(pixel_thresholds, pck_values, s = 100, color = 'darkblue')
plt.plot(pixel_thresholds, pck_values, color = 'blue')
plt.xlabel("Pixel Threshold")
plt.ylabel("PCK, %")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(pixel_thresholds)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "pck_curve_with_errorbars.png"))
plt.show()