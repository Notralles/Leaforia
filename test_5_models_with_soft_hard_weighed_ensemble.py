import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

# Dataset yolu
dataset_dir = r"C:\dataset"
test_dir = os.path.join(dataset_dir, "test")

# Sınıf isimleri
class_names = [
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy'
]
num_classes = len(class_names)

# Transformlar
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Test dataloader
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Model yükleyiciler
def load_model(name, path):
    if name == "mobilenetv2":
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnetb0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mnasnet":
        model = models.mnasnet1_0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "squeezenet":
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    elif name == "shufflenet":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Bilinmeyen model ismi")

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)


# Modelleri yükle
models_dict = {
    "mobilenetv2": load_model("mobilenetv2", "mobilenetv2.pth"),
    "efficientnetb0": load_model("efficientnetb0", "efficientnetb0.pth"),
    "mnasnet": load_model("mnasnet", "mnasnet.pth"),
    "squeezenet": load_model("squeezenet", "squeezenet.pth"),
    "shufflenet": load_model("shufflenet", "shufflenet.pth"),
}


# Ensemble değerlendirme
def evaluate_soft_voting(models_dict):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            avg_probs = None

            for model in models_dict.values():
                outputs = model(inputs)
                probs = nn.functional.softmax(outputs, dim=1)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            avg_probs /= len(models_dict)
            preds = torch.argmax(avg_probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# Metrik ve CM hesapla
def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))

    # Confusion matrix görseli
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["c0", "c1", "c2", "c3"],
                yticklabels=["c0", "c1", "c2", "c3"])
    plt.title("Soft Voting Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("soft_voting_confusion_matrix.png")
    plt.close()


# Değerlendir
y_true, y_pred = evaluate_soft_voting(models_dict)
evaluate_metrics(y_true, y_pred)
