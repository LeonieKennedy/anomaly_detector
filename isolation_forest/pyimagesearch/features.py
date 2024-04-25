from imutils import paths
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name).to(device)


def quantify_image(image, bins=(4, 6, 3)):
    # compute a 3D color histogram over the image and normalize it
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # return the histogram

    return hist


def load_dataset(datasetPath, bins):
    # grab the paths to all images in our dataset directory, then initialize our lists of images
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    # loop over the image paths
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for imagePath in imagePaths:
        # load the image and convert it to the HSV color space
        image = cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # quantify the image and update the data list
        inputs = clip_processor(text=["normal"], images=image, return_tensors="pt", padding=True).to(device)
        print(inputs)
        inputs.pop("input_ids")
        inputs.pop("attention_mask")

        image_embeddings = clip_model.get_image_features(**inputs)

            # outputs = clip_model(**inputs)
    #         print(f"""INPUT:
    # {inputs}
    #
    # OUTPUT:
    # {outputs}""")
        print(image_embeddings)
    #         print(outputs["image_embeds"].detach().numpy())
        # features = quantify_image(image, bins)
        # print(features)
        # data.append(image_embeddings)
        data.append(image_embeddings.cpu().detach().numpy()[0])
        # return our data list as a NumPy array

    return np.array(data)
    # return image_embeddings.cpu().detach().numpy()