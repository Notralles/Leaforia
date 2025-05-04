import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import (
    mobilenet_v2, resnet18, densenet121,
    shufflenet_v2_x1_0, efficientnet_b0
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
batch_size = 64
class_names = ['c0', 'c1', 'c2', 'c3']

# === Data ===
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

# === Model Loaders ===
def load_model(model_fn, path, num_classes):
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

# === Load Models ===
models = [
    load_model(init_mobilenetv2, "mobilenetv2_best.pth", num_classes),
    load_model(init_resnet18, "resnet18_best.pth", num_classes),
    load_model(init_densenet121, "densenet121_best.pth", num_classes),
    load_model(init_shufflenet, "shufflenetv2_best.pth", num_classes),
    load_model(init_efficientnetb0, "efficientnetb0_best.pth", num_classes),
]

# === Hard Voting Ensemble ===
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        batch_preds = []

        for model in models:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            batch_preds.append(preds.cpu().numpy())

        # Hard voting: majority vote (most frequent prediction)
        batch_preds = np.array(batch_preds).T  # Transpose to shape (batch_size, num_models)
        majority_vote = [Counter(preds).most_common(1)[0][0] for preds in batch_preds]
        all_preds.extend(majority_vote)
        all_labels.extend(labels.numpy())

# === Evaluation ===
acc = accuracy_score(all_labels, all_preds)
print(f"Hard Voting Ensemble Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

# === Confusion Matrix ===
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.5)  # Büyük punto
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("True", fontsize=16)
plt.title("Hard Voting Confusion Matrix", fontsize=18)
plt.tight_layout()
plt.savefig("hard_voting_confusion_matrix.png")
plt.show()
