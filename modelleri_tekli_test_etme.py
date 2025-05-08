import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import squeezenet1_0, shufflenet_v2_x1_0
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Klasör yolları
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

# Dönüşümler
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model yükleyiciler
def load_mobilenetv2():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv2.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_efficientnetb0():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("efficientnetb0.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_mnasnet():
    model = models.mnasnet1_0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mnasnet.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_squeezenet():
    model = squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    model.load_state_dict(torch.load("squeezenet.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_shufflenet():
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("shufflenet.pth", map_location=device))
    model.eval()
    return model.to(device)

# Değerlendirme fonksiyonu
def evaluate_model(model):
    model.eval()
    preds, probs, labels_all = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            softmax_probs = nn.functional.softmax(outputs, dim=1)
            _, pred_classes = torch.max(softmax_probs, 1)
            preds.extend(pred_classes.cpu().numpy())
            probs.extend(softmax_probs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return np.array(labels_all), np.array(preds)

# Confusion matrix çizimi
def plot_conf_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["c0", "c1", "c2", "c3"],
                yticklabels=["c0", "c1", "c2", "c3"])
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# Metrik hesaplayıcı
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "cm": confusion_matrix(y_true, y_pred)
    }

# Ana blok
if __name__ == "__main__":
    model_loaders = {
        "MobileNetV2": load_mobilenetv2,
        "EfficientNetB0": load_efficientnetb0,
        "MNASNet": load_mnasnet,
        "SqueezeNet": load_squeezenet,
        "ShuffleNet": load_shufflenet
    }

    for name, loader in model_loaders.items():
        print(f"\n------ {name} Sonuçları ------")
        model = loader()
        y_true, y_pred = evaluate_model(model)
        metrics = compute_metrics(y_true, y_pred)

        print(classification_report(y_true, y_pred, target_names=class_names))
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        plot_conf_matrix(metrics["cm"], f"{name} Confusion Matrix")
