import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.optim import Adam
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
train_path = '../Crime_Dataset'


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.label = clip.tokenize(
            list_txt)  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        label = self.label[idx]
        return image, label


# use your own data
directories = os.listdir(train_path)
list_image_path = []
list_txt = []

for directory in directories:
    for file in os.listdir(os.path.join(train_path, directory)):
        list_image_path.append(train_path + "/" + directory + "/" + file)
        list_txt.append(directory)

# print(f"""list_txt: {list_txt}
# list_image_path: {list_image_path}""")

dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size=32)  # Define your own dataloader


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
for epoch in tqdm(range(5)):
    for i, batch in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)