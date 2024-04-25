import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
import os

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name).to(device)


# Function to extract frames from video
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
    cap.release()
    return frames


# Function to compute reference features from videos
def compute_reference_features(video_folder, num_frames_per_video=10):
    all_features = []
    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)
        frames = extract_frames(video_path, num_frames_per_video)
        for frame in frames:
            inputs = clip_processor(images=frame)
            pixel_values = torch.tensor(np.array(inputs["pixel_values"]))
            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=pixel_values)
                all_features.append([image_features])
    print(image_features)

    reference_features = np.mean(all_features, axis=0)  # Aggregate features by taking mean
    reference_features = reference_features / np.linalg.norm(reference_features)  # Normalize features
    return reference_features


# Define anomaly detection threshold
threshold = 0.8


# Function to detect anomalies in video
def detect_anomaly_video(video_path):
    reference_features = compute_reference_features("C:/Users/Student/Downloads/UCF101_subset/train/Basketball")
    frames = extract_frames(video_path)
    anomaly_count = 0
    for idx, frame in enumerate(frames):
        inputs = clip_processor(text=["a photo of a normal scene"], images=frame, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_features = outputs
            similarity_score = torch.cosine_similarity(image_features, reference_features)
        if similarity_score < threshold:
            print(f"Anomaly detected in frame {idx} of {video_path}")
            anomaly_count += 1
    if anomaly_count == 0:
        print(f"No anomalies detected in {video_path}")


# Test anomaly detection on a video
video_path = "C:/Users/Student/Downloads/UCF101_subset/train/Archery/v_Archery_g05_c04.avi"
detect_anomaly_video(video_path)
