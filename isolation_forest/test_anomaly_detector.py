# import the necessary packages
from pyimagesearch.features import quantify_image
import argparse
import pickle
import cv2
from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name).to(device)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained anomaly detection model")
ap.add_argument("-i", "--image", required=True,	help="path to input image")
args = vars(ap.parse_args())

# load the anomaly detection model
print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())
# load the input image, convert it to the HSV color space, and
# quantify the image in the *same manner* as we did during training
image = cv2.imread(args["image"])
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# features = quantify_image(hsv, bins=(3, 3, 3))
inputs = clip_processor(text=["normal"], images=image, return_tensors="pt", padding=True)
features = clip_model(**inputs)
# use the anomaly detector model and extracted features to determine
# if the example image is an anomaly or not
preds = model.predict([features["image_embeds"].detach().numpy()[0]])[0]
# preds = model.predict([features])[0]
label = "anomaly" if preds == -1 else "normal"
color = (0, 0, 255) if preds == -1 else (0, 255, 0)
# draw the predicted label text on the original image
cv2.putText(image, label, (10,  25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
# display the image
cv2.imshow("Output", image)
cv2.waitKey(0)
