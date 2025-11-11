# Pawtellige: Dog Behavior Recognition System

Pawtellige is an AI-powered system that detects and analyzes dog behaviors using deep learning.  
It helps pet owners and researchers monitor dog activities, identify abnormal behaviors,  
and understand emotional or health conditions through real-time image and video analysis.

---

## Features

- Real-time pose recognition from images or videos  
- Deep learning models: ResNet50 and EfficientNet support  
- Behavior logging using JSON for time-based tracking  
- Grad-CAM++ visualization to interpret model attention areas  
- Abnormal activity detection (e.g., trembling, limping)  
- Interactive dashboard for visualization and monitoring  

---

## Project Structure
├── abnormal_detector.py # Detects abnormal dog behavior
├── alarm_sender.py # Sends notifications or alerts
├── behavior_logger.py # Logs recognized behaviors into JSON
├── dog_dashboard.py # Dashboard visualization
├── dog_monitor_log.json # Behavior log file
├── dog_pose_classes.txt # Behavior/pose label list
├── dog_pose_efficientnet.py # EfficientNet model inference
├── dog_pose_resnet50.py # ResNet50 model inference
├── gradcam_plus_plus.py # Grad-CAM++ visualization tool
├── main_monitor.py # Main script for real-time monitoring
├── predict_dog_pose.py # Predicts dog pose from an image
├── train_dog_pose.py # Model training script
├── train_efficientnet.py # EfficientNet training script
├── test_video.py # Video test script
├── test_image.jpg # Example test image
├── input_video.mp4 # Example input video
└── README.md # Project documentation

## Installation

### 1. Clone the repository
git clone https://github.com/Christalwu1110/pawtellige.git
cd pawtellige

2. Install dependencies

If you do not have a requirements.txt, install them manually:

pip install torch torchvision opencv-python numpy matplotlib

How to Use
1. Train a model
python train_dog_pose.py

2. Run a single image prediction
python predict_dog_pose.py --image test_image.jpg

3. Start real-time monitoring
python main_monitor.py

Example Outputs
Function	Output
Real-time Detection	Bounding boxes and pose labels
Grad-CAM++	Heatmap visualization
Logs	Saved in dog_monitor_log.json
Dashboard	Real-time updates via dog_dashboard.py
Model Information

ResNet50 model: dog_pose_resnet50.pth (~90 MB)

EfficientNet model: optional alternative

Large model files (>50 MB) are not recommended to upload directly to GitHub.
You can host them externally (for example, Google Drive or Dropbox) and provide a link below:
Download pretrained model: https://drive.google.com/your-model-link


Place the file in the project root directory before running predictions.

Future Improvements

Multi-dog detection and tracking

Edge deployment (ESP32-CAM or Raspberry Pi)

Emotion recognition (pain, fear, excitement)

Web-based monitoring dashboard

Author
Christal Wu
Email: yeling031110@163.com
GitHub: https://github.com/Christalwu1110

Tech Stack
Category	Tools
Language	Python 3.10+
Deep Learning	PyTorch
Computer Vision	OpenCV
Visualization	Matplotlib, Grad-CAM++
Data Logging	JSON
Dashboard	Local visualization
