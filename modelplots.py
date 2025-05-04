import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_classes = 4
class_names = ['c0', 'c1', 'c2', 'c3']

# === Test Dataset ===
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dir = r"C:\dataset\test"
test_dataset = ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Model Initializers ===
from torchvision.models import (
    mobilenet_v2, resnet18, densenet121,
    shufflenet_v2_x1_0, efficientnet_b0
)

def init_and_load_model(model_fn, path, num_classes):
    model = model_fn(num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def init_mobilenetv2(num_classes):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    return model

def init_resnet18(num_classes):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def init_densenet121(num_classes):
    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model

def init_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def init_efficientnetb0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# === Model List ===
model_paths = [
    ("MobileNetV2", init_mobilenetv2, "mobilenetv2_best.pth"),
    ("ResNet18", init_resnet18, "resnet18_best.pth"),
    ("DenseNet121", init_densenet121, "densenet121_best.pth"),
    ("ShuffleNetV2", init_shufflenet, "shufflenetv2_best.pth"),
    ("EfficientNetB0", init_efficientnetb0, "efficientnetb0_best.pth")
]

# === Prediction & Plotting Function ===
def plot_confusion_matrix(model_name, model):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.5)  # Büyük punto
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix", fontsize=18)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_confusion_matrix.png")
    plt.show()

# === Plot Confusion Matrices for All Models ===
for name, init_fn, path in model_paths:
    print(f"Generating confusion matrix for {name}...")
    model = init_and_load_model(init_fn, path, num_classes)
    plot_confusion_matrix(name, model)
