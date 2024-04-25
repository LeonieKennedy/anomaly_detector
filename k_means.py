import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import os
# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example dataset (replace with your own dataset)
images = os.listdir("C:/Users/Student/Downloads/brains/Data/Normal/")

# Extract features using CLIP
image_features = []
text_features = []

for image_path in images:
    description = "Normal"
    image = Image.open(("C:/Users/Student/Downloads/brains/Data/Normal/"+image_path))
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    # with torch.no_grad():
    #     output = clip_model(**inputs)
    #     print(output)
    image_embeddings = clip_model.get_image_features(**inputs)
    image_features.append(image_embeddings)
    inputs = clip_processor(text="Normal", return_tensors="pt", padding=True)
    text_embeddings = clip_model.get_text_features(**inputs)
    text_features.append(text_embeddings)
# Concatenate image and text features
features = np.concatenate([image_features, text_features], axis=1)

# Apply K-means clustering
kmeans = KMeans(n_clusters=1)  # Assuming there are normal data and anomalies
kmeans.fit(features)

# Save the CLIP model
clip_model.save_pretrained("saved_models/clip_model")

# Save the K-means model
joblib.dump(kmeans, "saved_models/kmeans_model.pkl")

# Save other necessary components (e.g., image processor, scaler)
# Make sure to save any preprocessing steps required for inference
# For instance, if you're using image resizing or normalization, save them as well
# This ensures consistency during inference

print("Models saved successfully!")

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import joblib

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("saved_models/clip_model").to(device)
clip_processor = CLIPProcessor.from_pretrained("saved_models/clip_model")

# Load K-means model
kmeans = joblib.load("saved_models/kmeans_model.pkl")

# Example image (replace with your own image path)
image_path = "C:/Users/Student/Downloads/brains/Data/Tumor/meningioma_tumor/M_9_RO_.jpg"
description = "Tumour"

# Preprocess the image
image = Image.open(image_path)

# Extract features using CLIP
inputs = clip_processor(text=[description], images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    output = clip_model(**inputs)
    print(output)
image_embeddings = clip_model.get_image_features(**inputs)
text_features = clip_model.get_text_features(**inputs)

# Concatenate image and text features
features = np.concatenate([image_features, text_features], axis=0)

# Predict cluster
cluster = kmeans.predict([features])[0]

# Check if the image is an anomaly or normal
if cluster == 1:
    print("Anomaly detected!")
else:
    print("Normal image.")

# Optionally, you can also print the anomaly score if needed:
anomaly_score = np.linalg.norm(features - kmeans.cluster_centers_[cluster])
print("Anomaly score:", anomaly_score)
