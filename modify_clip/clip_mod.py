import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam
from transformers import CLIPProcessor, CLIPModel

# Define dataset paths
train_path = '../Crime_Dataset'

# Define batch size
batch_size = 32
validation_split = 0.2

# Define CLIP model
# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)

# Modify the final classification layer
num_classes = 8  # normal vs. anomaly
# model.classifier = torch.nn.Linear(model.visual.output_dim, num_classes).to(device)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        sample = processor(images=sample, return_tensors="pt").to(device)
        return sample, target

# Load dataset
train_dataset = CustomImageFolder(root=train_path)
print(train_dataset[0])
# Create ImageFolder dataset
# train_dataset = ImageFolder(root=train_path, transform=transform)

# Calculate the split indices for validation set
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(validation_split * num_train)
train_indices = indices[split:]
val_indices = indices[:split]

# Create Samplers for train and validation sets
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create DataLoader for training data
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

# Check the class names
class_names = train_dataset.classes
print("Class names:", class_names)

# # Load UCF Crime dataset
# train_dataset = UCF101(root=train_path, annotation_path='ucfTrainTestlist/trainlist01.txt', frames_per_clip=16,
#                        step_between_clips=1, frame_rate=None, fold=1, train=True, transform=transform)
# val_dataset = UCF101(root=val_path, annotation_path='ucfTrainTestlist/testlist01.txt', frames_per_clip=16,
#                      step_between_clips=1, frame_rate=None, fold=1, train=False, transform=transform)

# Create data loaders
# train_loader = ImageFolder(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = ImageFolder(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataset:
        # print("input:" , inputs, "\n\n", "labels:", labels)
        # labels = processor(class_names, return_tensors="pt", padding=True)
        print(f"""inputs:
        {inputs}
        labels:
        {labels}""")
        # print(len(labels), len(inputs))
        # inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs)
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%")
