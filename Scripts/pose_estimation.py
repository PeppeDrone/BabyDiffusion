import os
import torch
import numpy as np
from PIL import Image
import cv2
import json
from collections import defaultdict, Counter
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO
from tqdm import tqdm
from rtmlib import Wholebody, draw_skeleton

# Set device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Flag for full body pose estimation
USE_FULL_BODY = True  # Set to False for simple pose estimation

# Initialize RTMLib Wholebody model if needed
if USE_FULL_BODY:
    print("Initializing RTMLib Wholebody model...")
    wholebody = Wholebody(
        to_openpose=False,
        mode='balanced',  # Using balanced mode for best accuracy/speed tradeoff
        backend='onnxruntime',
        device=device
    )
else:
    # Load ViT Pose model and processor
    pose_model_name = "usyd-community/vitpose-base-simple"
    print(f"Loading ViT Pose model: {pose_model_name}...")
    processor = AutoProcessor.from_pretrained(pose_model_name)
    pose_model = VitPoseForPoseEstimation.from_pretrained(pose_model_name).to(device)
    pose_model.eval()

# Load YOLOv8 model for person detection
print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8s.pt").to('cpu')

# Define dataset paths
dataset_dir = "mydataset_local/train"
images_dir = os.path.join(dataset_dir, "images")
train2_dir = "mydataset_local/train4"
train2_images_dir = os.path.join(train2_dir, "images")
train2_poses_dir = os.path.join(train2_dir, "poses")
os.makedirs(train2_images_dir, exist_ok=True)
os.makedirs(train2_poses_dir, exist_ok=True)

# Define skeleton connections based on model type
if USE_FULL_BODY:
    pass # already defined in RTMLib
    # # Full body skeleton connections (COCO format with additional keypoints)
    # skeleton_connections = [
    #     (15, 13), (13, 11),        # left leg: ankle->knee, knee->hip
    #     (16, 14), (14, 12),        # right leg: ankle->knee, knee->hip
    #     (11, 12),                  # pelvis: left hip->right hip
    #     (5, 11), (6, 12), (5, 6),  # torso: left shoulder->left hip, right shoulder->right hip, shoulder->shoulder
    #     (5, 7), (7, 9),            # left arm: shoulder->elbow, elbow->wrist
    #     (6, 8), (8, 10),           # right arm: shoulder->elbow, elbow->wrist
    #     (1, 2), (0, 1), (0, 2),    # head: eyes->each other and nose->eyes
    #     (1, 3), (2, 4),            # face: left eye->left ear, right eye->right ear
    #     (3, 5), (4, 6),            # neck: left ear->left shoulder, right ear->right shoulder
    #     (17, 15), (18, 16),        # feet: left foot->left ankle, right foot->right ankle
    #     (19, 17), (20, 18)         # toes: left toe->left foot, right toe->right foot
    # ]
    
    # # Define colors for different body parts (BGR format for OpenCV)
    # color_map = {
    #     'left_leg':   (0, 255, 0),    # green
    #     'right_leg':  (255, 0, 0),    # blue
    #     'torso':      (255, 255, 255),# white
    #     'left_arm':   (0, 255, 255),  # yellow
    #     'right_arm':  (255, 0, 255),  # magenta
    #     'face':       (0, 0, 255),    # red
    #     'feet':       (128, 0, 128),  # purple
    #     'toes':       (0, 128, 128)   # teal
    # }
    
    # # Assign body part groups to connections
    # connection_groups = [
    #     'left_leg', 'left_leg',
    #     'right_leg','right_leg',
    #     'torso',
    #     'torso','torso','torso',
    #     'left_arm','left_arm',
    #     'right_arm','right_arm',
    #     'face','face','face',
    #     'face','face',
    #     'torso','torso',
    #     'feet','feet',
    #     'toes','toes'
    # ]
else:
    # Simple skeleton connections (original code)
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

# Confidence threshold for keypoints
keypoint_score_thresh = 0.5

# Process images
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
kept_images = set()

for img_name in tqdm(image_files, desc="Processing images"):
    # Check if files already exist
    pose_name = img_name[:-4]+'_pose.png'
    out_pose_path = os.path.join(train2_poses_dir, pose_name)
    out_img_path = os.path.join(train2_images_dir, img_name)
    
    if os.path.exists(out_pose_path) and os.path.exists(out_img_path):
        print(f"Skipping {img_name} - already processed")
        kept_images.add(img_name)
        continue
        
    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    
    # Detect person
    results = yolo_model(image, classes=[0])
    if len(results) == 0 or len(results[0].boxes) != 1:
        continue
        
    # Get person bounding box
    xyxy = results[0].boxes.xyxy[0].cpu().numpy()
    person_boxes = np.array([xyxy])
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
    
    if USE_FULL_BODY:
        # Convert PIL image to OpenCV format for RTMLib
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Perform RTMLib inference
        keypoints, scores = wholebody(cv_image)
        if keypoints.shape[1] < 133: 
            continue
        
        # Create skeleton image
        skeleton_img = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw skeleton using RTMLib's draw_skeleton function
        skeleton_img = draw_skeleton(skeleton_img, keypoints, scores, kpt_thr=keypoint_score_thresh)
    else:
        # Process pose with ViT Pose
        inputs = processor(image, boxes=[person_boxes], return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = v.to(device)
            
        dataset_index = torch.tensor([5], device=device)
        with torch.no_grad():
            outputs = pose_model(**inputs, dataset_index=dataset_index)
            
        pose_results = processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
        pose = pose_results[0][0]
        keypoints = pose['keypoints'].cpu().numpy()
        scores = pose['scores'].cpu().numpy()
            
        # Count valid keypoints
        num_keypoints = int(np.sum(scores >= keypoint_score_thresh))
        if num_keypoints < 17:  # Skip if not all keypoints are detected
            continue
        
        # Draw pose
        skeleton_img = np.zeros((height, width, 3), dtype=np.uint8)
        for (idx, (i, j)) in enumerate(skeleton_connections):
            pt1 = tuple(int(v) for v in keypoints[i])
            pt2 = tuple(int(v) for v in keypoints[j])
            conf1 = scores[i]
            conf2 = scores[j]
            if conf1 < keypoint_score_thresh or conf2 < keypoint_score_thresh:
                continue
            group = connection_groups[idx]
            color = color_map[group]
            cv2.line(skeleton_img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
        
        for k, (x, y) in enumerate(keypoints):
            if scores[k] < keypoint_score_thresh:
                continue
            cv2.circle(skeleton_img, (int(x), int(y)), radius=3, color=(0,0,255), thickness=-1)
    
    # Save results
    kept_images.add(img_name)
    pose_name = img_name[:-4]+'_pose.png'
    out_pose_path = os.path.join(train2_poses_dir, pose_name)
    out_img_path = os.path.join(train2_images_dir, img_name)
    cv2.imwrite(out_pose_path, skeleton_img)
    image.save(out_img_path)

# Save filtered prompt.json for kept images
prompt_in_path = os.path.join(dataset_dir, "prompt.json")
prompt_out_path = os.path.join(train2_dir, "prompt.json")
with open(prompt_in_path, "r", encoding="utf-8") as infile, open(prompt_out_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        row = json.loads(line)
        if row["target"] in kept_images:
            outfile.write(json.dumps(row) + "\n")
print(f"Filtered prompt.json saved to: {prompt_out_path}")

print("\nPose estimation completed!") 