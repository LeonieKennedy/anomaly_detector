import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")# Choose computation device

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

def get_training_data(directory):
    class_list = []
    training_data = []

    for classname in os.walk(directory):
        for image in os.walk(classname):

