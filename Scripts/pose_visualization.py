import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from transformers import AutoProcessor, VitPoseForPoseEstimation
from ultralytics import YOLO

def visualize_pose_estimation(image_path):
    # Set device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load ViT Pose model and processor
    pose_model_name = "usyd-community/vitpose-base-simple"
    print("Loading ViT Pose model...")
    processor = AutoProcessor.from_pretrained(pose_model_name)
    pose_model = VitPoseForPoseEstimation.from_pretrained(pose_model_name).to(device)
    pose_model.eval()

    # Load YOLOv8 model for person detection
    print("Loading YOLOv8 model...")
    yolo_model = YOLO("yolov8s.pt").to('cpu')

    # Define skeleton connections (COCO format)
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
    keypoint_score_thresh = 0.3

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Detect person
    results = yolo_model(image, classes=[0])
    if len(results) == 0 or len(results[0].boxes) != 1:
        print("No person detected in the image!")
        plt.close()
        return
        
    # Get person bounding box
    xyxy = results[0].boxes.xyxy[0].cpu().numpy()
    person_boxes = np.array([xyxy])
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
    
    # Plot 2: Image with bounding box
    ax2.imshow(image)
    x1, y1, x2, y2 = xyxy
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
    ax2.add_patch(rect)
    ax2.set_title('Image with Bounding Box')
    ax2.axis('off')
    
    # Process pose
    inputs = processor(image, boxes=[person_boxes], return_tensors="pt")
    for k,v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = pose_model(**inputs)
    pose_results = processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    pose = pose_results[0][0]
    keypoints = pose['keypoints'].cpu().numpy()
    scores = pose['scores'].cpu().numpy()
    
    # Plot 3: Image with skeleton
    ax3.imshow(image)
    for (idx, (i, j)) in enumerate(skeleton_connections):
        pt1 = tuple(int(v) for v in keypoints[i])
        pt2 = tuple(int(v) for v in keypoints[j])
        conf1 = scores[i]
        conf2 = scores[j]
        if conf1 < keypoint_score_thresh or conf2 < keypoint_score_thresh:
            continue
        group = connection_groups[idx]
        color = tuple(c/255 for c in color_map[group])  # Convert BGR to RGB and normalize
        ax3.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=color, linewidth=2)
    
    for k, (x, y) in enumerate(keypoints):
        if scores[k] < keypoint_score_thresh:
            continue
        ax3.plot(x, y, 'ro', markersize=3)
    
    ax3.set_title('Image with Skeleton')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with your image path
    image_path = "mydataset_local/train/images/segment_0893_029_frame_000150.jpg"
    visualize_pose_estimation(image_path) 