from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize images to a consistent size
            img_data = np.array(img)
            img_data_flat = img_data.flatten()  # Flatten the image
            images.append(img_data_flat)
    return np.array(images)

# Directory containing images
image_folder = "C:/Users/Student/Downloads/forest/Data/Forest"

# Load images and create feature matrix X
X = load_images_from_folder(image_folder)

# Normalize pixel values to range [0, 1]
X = X / 255.0

# Print shape of feature matrix X
print("Shape of feature matrix X:", X.shape)


# Train Isolation Forest model
model = IsolationForest(contamination=0.01, random_state=42)  # Adjust contamination based on anomaly rate
model.fit(X)

img = Image.open("C:/Users/Student/Downloads/forest/Data/Forest/forest.93.jpg").convert('L')  # Convert to grayscale
img = img.resize((64, 64))  # Resize images to a consistent size
img_data = np.array(img)
img_data_flat = img_data.flatten().reshape(1, -1)  # Flatten the image

# Predict anomalies on test data
y_pred = model.predict(img_data_flat)[0]
print(y_pred)
# Convert predictions to binary labels (1 for anomalies, -1 for normal)
label = "anomaly" if y_pred == -1 else "normal"
print(label)
