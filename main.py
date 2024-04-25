import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.to(device)


def detect_objects(image_path):
    labels = ["arson", "assault", "theft", "normal"]
    image = Image.open(image_path)
    inputs = processor(text=labels, images=image, return_tensors="pt")
    inputs.to(device)

    output = model(**inputs)

    logits_per_image = output.logits_per_image
    probs = logits_per_image.softmax(dim=-1).detach().numpy()[0]

    for count, i in enumerate(probs):
        print(f"label: {labels[count]}, probability: {i}")


image_path = "C:/Users/Student/Documents/University_of_Surrey/Dissertation/Project/normal3.jpg"
detect_objects(image_path)
