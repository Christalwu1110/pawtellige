import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import os
import json
import time
import datetime
import matplotlib.pyplot as plt
from behavior_logger import DogBehaviorLogger
from abnormal_detector import AbnormalBehaviorDetector

# --- Configuration ---
MODEL_TYPE = "efficientnet-b0"
MODEL_PATH = "/home/yeling/dog_project/dog_pose_efficientnet_b0.pth"
CLASSES_PATH = "/home/yeling/dog_project/dog_pose_classes.txt"
VIDEO_PATH = "/home/yeling/dog_project/input_video.mp4"
OUTPUT_DIR = "/home/yeling/dog_project/key_frames"
LOG_PATH = "/home/yeling/dog_project/dog_monitor_log.json"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FRAME_THRESHOLD = 1000  # For static frames
FRAME_INTERVAL = 3  # Every 3rd frame
UNIFORM_INTERVAL = 30  # Every 1s
UPDATE_INTERVAL = 50  # Update logger every 50 frames

# --- Validate Paths ---
def validate_paths():
    for path in [MODEL_PATH, CLASSES_PATH, VIDEO_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"All paths validated: {MODEL_PATH}, {CLASSES_PATH}, {VIDEO_PATH}, {OUTPUT_DIR}")

# --- Validate Video ---
def validate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video validated: {video_path}, {frame_count} frames, {fps} fps")
    return fps, frame_count

# --- Load Classes ---
def load_classes():
    try:
        with open(CLASSES_PATH, 'r') as f:
            class_map = json.load(f)
        class_names = list(class_map.keys())
        if not class_names:
            raise ValueError("No classes found in dog_pose_classes.txt")
        print(f"Classes loaded: {class_names}")
        return class_names
    except Exception as e:
        print(f"Error loading classes: {e}")
        exit(1)

# --- Initialize Model ---
def init_model(class_names):
    try:
        model = EfficientNet.from_name(MODEL_TYPE)
        model._fc = torch.nn.Linear(model._fc.in_features, len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("Model initialized: EfficientNet-B0")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# --- Data Transform ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Frame Extraction ---
def extract_keyframes(video_path, output_dir, threshold=FRAME_THRESHOLD, interval=FRAME_INTERVAL, uniform_interval=UNIFORM_INTERVAL):
    fps, frame_count = validate_video(video_path)
    cap = cv2.VideoCapture(video_path)
    keyframes = []
    prev_frame = None
    frame_count = 0
    keyframe_indices = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame differencing
        if frame_count % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                diff_sum = np.sum(diff)
                if diff_sum > threshold:
                    frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    keyframes.append((frame_count, frame_path, frame, frame_count / fps))
                    keyframe_indices.append(frame_count)
            prev_frame = gray
        
        # Uniform sampling
        if frame_count % uniform_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            keyframes.append((frame_count, frame_path, frame, frame_count / fps))
            keyframe_indices.append(frame_count)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(keyframes)} keyframes")
    return keyframes, keyframe_indices, fps

# --- Pose Prediction ---
def predict_pose(model, frame, transform, device):
    frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(frame)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        pose = class_names[pred_idx.item()]
    return pose, confidence.item()

# --- Main Processing ---
def main():
    start_time = time.time()
    
    # Validate paths
    validate_paths()
    
    # Initialize logger and detector
    logger = DogBehaviorLogger(log_file=LOG_PATH)
    detector = AbnormalBehaviorDetector(logger)
    
    # Clear previous log
    logger.clear_log()
    
    # Load classes and model
    global class_names
    class_names = load_classes()
    model = init_model(class_names)
    
    # Extract keyframes
    keyframes, keyframe_indices, fps = extract_keyframes(VIDEO_PATH, OUTPUT_DIR)
    if not keyframes:
        print("Error: No keyframes extracted. Exiting.")
        return
    
    # Predict poses and check abnormalities
    pose_counts = {'active': 0, 'eating': 0, 'lying': 0, 'sitting': 0, 'standing': 0}
    timestamps = []
    poses = []
    
    for i, (frame_idx, frame_path, frame, frame_time) in enumerate(keyframes):
        # Predict pose
        pose, confidence = predict_pose(model, frame, transform, DEVICE)
        
        # Convert frame time to ISO format
        video_start = datetime.datetime(2025, 9, 1, 4, 36, 0)  # Fixed start for consistency
        timestamp = (video_start + datetime.timedelta(seconds=frame_time)).isoformat()
        
        # Log behavior
        logger.add_behavior_record(pose, confidence, timestamp, frame_idx, frame_path)
        pose_counts[pose] += 1
        timestamps.append(timestamp)
        poses.append(pose)
        
        # Check abnormalities every UPDATE_INTERVAL frames
        if i == 0 or (i + 1) % UPDATE_INTERVAL == 0 or i == len(keyframes) - 1:
            detector.check_for_abnormalities(current_check_time=timestamp)
    
    # Save pose distribution plot
    plt.figure(figsize=(6, 6))
    plt.pie(pose_counts.values(), labels=pose_counts.keys(), autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title('EfficientNet-B0 Pose Distribution')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pose_distribution_effnet.png'))
    plt.close()
    
    # Save timeline plot
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, poses, marker='o', color='#1f77b4')
    plt.xlabel('Time')
    plt.ylabel('Pose')
    plt.title('EfficientNet-B0 Pose Timeline')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pose_timeline_effnet.png'))
    plt.close()
    
    # Extract and save alerts
    records = logger.get_records()
    alerts = [f"{r['abnormal_flags']}" for r in records if r['abnormal_flags']]
    with open(os.path.join(OUTPUT_DIR, 'alerts.txt'), 'w') as f:
        for alert in alerts:
            f.write(f"{alert}\n")
    
    print(f"Processing complete in {time.time() - start_time:.2f} seconds")
    print(f"Pose distribution: {pose_counts}")
    print(f"Alerts: {alerts}")
    print(f"Visualizations saved to {OUTPUT_DIR}")
    print(f"Start Dash dashboard at http://127.0.0.1:8050")

if __name__ == "__main__":
    main()
    