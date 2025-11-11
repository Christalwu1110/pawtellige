import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import os
import json
import time
import datetime
from behavior_logger import DogBehaviorLogger
from abnormal_detector import AbnormalBehaviorDetector

# --- Configuration ---
MODEL_TYPE = "efficientnet-b0"
MODEL_PATH = "/home/yeling/dog_project/dog_pose_efficientnet_b0.pth"
CLASSES_PATH = "/home/yeling/dog_project/dog_pose_classes.txt"
VIDEO_SOURCE = "/home/yeling/dog_project/input_video.mp4"  # Path to 50-second video
OUTPUT_DIR = "/home/yeling/dog_project/key_frames"
LOG_PATH = "/home/yeling/dog_project/dog_monitor_log.json"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FRAME_THRESHOLD = 1000
FRAME_INTERVAL = 10  # Process keyframe every 10 frames
UNIFORM_INTERVAL = 30
UPDATE_INTERVAL = 50

# --- Validate Paths ---
def validate_paths():
    for path in [MODEL_PATH, CLASSES_PATH, VIDEO_SOURCE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"All paths validated: {MODEL_PATH}, {CLASSES_PATH}, {VIDEO_SOURCE}, {OUTPUT_DIR}")

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

# --- Process Video Stream ---
def process_video_stream(video_source, output_dir, logger, detector, model, class_names, threshold=FRAME_THRESHOLD, interval=FRAME_INTERVAL, uniform_interval=UNIFORM_INTERVAL):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_source}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    prev_frame = None
    video_start = datetime.datetime.now()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Video stream ended: {video_source}")
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
                    process_frame(frame, frame_path, frame_count, video_start, fps, logger, detector, model, class_names)
            prev_frame = gray
        
        # Uniform sampling
        if frame_count % uniform_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            process_frame(frame, frame_path, frame_count, video_start, fps, logger, detector, model, class_names)
        
        frame_count += 1
        
        # Check abnormalities periodically
        if frame_count % UPDATE_INTERVAL == 0:
            timestamp = (video_start + datetime.timedelta(seconds=frame_count / fps)).isoformat()
            detector.check_for_abnormalities(current_check_time=timestamp)
        
        # Simulate real-time processing (optional: adjust for 50-second video)
        time.sleep(1.0 / fps)  # Control processing speed to match video frame rate
        
        # Allow interruption
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# --- Process Single Frame ---
def process_frame(frame, frame_path, frame_count, video_start, fps, logger, detector, model, class_names):
    pose, confidence = predict_pose(model, frame, transform, DEVICE)
    timestamp = (video_start + datetime.timedelta(seconds=frame_count / fps)).isoformat()
    logger.add_behavior_record(pose, confidence, timestamp, frame_count, frame_path)

# --- Predict Pose ---
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
    
    # Process video
    print(f"Starting video processing: {VIDEO_SOURCE}. Press 'q' to quit.")
    process_video_stream(VIDEO_SOURCE, OUTPUT_DIR, logger, detector, model, class_names)

if __name__ == "__main__":
    main()
    