import json
import os
from threading import Lock

class DogBehaviorLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.lock = Lock()
        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def add_behavior_record(self, pose, confidence, timestamp, frame_idx, frame_path):
        with self.lock:
            try:
                with open(self.log_file, 'r+', encoding='utf-8') as f:
                    logs = json.load(f)
                    logs.append({
                        'frame': frame_idx,
                        'path': frame_path,
                        'predicted_pose': pose,
                        'confidence': confidence,
                        'timestamp': timestamp,
                        'abnormal_flags': []
                    })
                    f.seek(0)
                    f.truncate()
                    json.dump(logs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error writing log: {e}")

    def get_records(self):
        with self.lock:
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading log: {e}")
                return []

    def clear_log(self):
        with self.lock:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
                