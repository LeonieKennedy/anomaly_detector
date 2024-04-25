import os
from prepare_video_dataset import extract_features
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

PATH_TO_DATASET_DIRECTORY = "C:/Users/Student/Downloads/UCF101_subset/train/Basketball/"

#
# def train_model(training_set):
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(training_set)
#
#     model = IsolationForest(n_estimators=100, contamination=0.2)
#     model.fit(features_scaled)
#
#     return model
#
#
# def iterate_through_normal(directory):
#     feature_list = []
#     for filename in os.listdir(directory):
#         video_file = os.path.join(directory, filename)
#         features = extract_features(video_file)
#
#         if len(features) > 0:
#             feature_list.extend(features)
#
#     return np.array(feature_list)
#
#
# if __name__ == "__main__":
#     training_set = iterate_through_normal(PATH_TO_DATASET_DIRECTORY)
#     model = train_model(training_set)
#
#     test_image_features = extract_features("C:/Users/Student/Downloads/UCF101_subset/test/Archery/v_Archery_g02_c07.avi")
#     prediction = model.predict(test_image_features)
#
#     print(f"result: {prediction}")

import cv2
import os
import numpy as np
from sklearn.ensemble import IsolationForest


def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            frame_resized = cv2.resize(frame_gray, (64, 64))  # Resize frame to a consistent size
            frame_data_flat = frame_resized.flatten()  # Flatten the frame
            frames.append(frame_data_flat)

    cap.release()
    return np.array(frames)


def load_videos_from_folder(folder):
    video_paths = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith('.avi')]
    print("Found videos:", video_paths)
    videos_data = []
    for video_path in video_paths:
        video_frames = extract_frames(video_path)
        print("Loaded frames from", video_path)
        videos_data.append(video_frames)
    return np.vstack(videos_data)


def train_isolation_forest(X):
    # Train Isolation Forest model
    model = IsolationForest(contamination=0.1, random_state=42)  # Adjust contamination based on anomaly rate
    model.fit(X)
    return model


def test_video(video_path, model):
    # Extract frames from the video
    frames = extract_frames(video_path)

    # Normalize pixel values to range [0, 1]
    frames = frames / 255.0

    # Predict anomalies using the trained model
    y_pred = model.predict(frames)

    # Convert predictions to binary labels (1 for anomalies, -1 for normal)
    y_pred_binary = np.where(y_pred == -1, 1, 0)

    # Print number of anomalies detected
    print("Number of anomalies detected in the video:", np.sum(y_pred_binary == 1))


# Directory containing videos
video_folder = PATH_TO_DATASET_DIRECTORY

# Load videos and create feature matrix X
X = load_videos_from_folder(video_folder)

# Normalize pixel values to range [0, 1]
X = X / 255.0

# Train Isolation Forest model
model = train_isolation_forest(X)

# Path to the single video file for testing
single_video_path = "C:/Users/Student/Downloads/UCF101_subset/test/Archery/v_Archery_g02_c07.avi"

# Test the video using the trained Isolation Forest model
test_video(single_video_path, model)
