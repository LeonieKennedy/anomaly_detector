import cv2
import numpy as np


def extract_features(video_file):
    cap = cv2.VideoCapture(video_file)
    feature_list = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from the frame (you can replace this with your own feature extraction method)
        features = extract_frame_features(frame)

        if features is not None:
            feature_list.append(features)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return np.array(feature_list)


def extract_frame_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    return hist.flatten()


# if __name__ == "__main__":
#     video_file = 'v_Basketball_g25_c02.avi'  # Replace with your video file path
#     features = extract_features(video_file)
#     print("Extracted features shape:", features.shape)
