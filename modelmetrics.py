import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.models import (
    mobilenet_v2, resnet18, densenet121,
    shufflenet_v2_x1_0, efficientnet_b0
)
import pandas as pd

# === Cihaz ve Sınıf Bilgisi ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['c0', 'c1', 'c2', 'c3']
num_classes = len(class_names)
batch_size = 64

# === Dönüştürme ve Test Verisi ===
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dir = r"C:\dataset\test"
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Model Yükleyici ===
def load_model(model_fn, path, classifier_attr):
    model = model_fn(weights=None)
    setattr(model, *classifier_attr)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === Model Bilgisi ve Yolları ===
model_info = {
    "mobilenetv2": {
        "fn": mobilenet_v2,
        "path": "mobilenetv2_best.pth",
        "classifier": ("classifier", torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1280, num_classes)
        ))
    },
    "resnet18": {
        "fn": resnet18,
        "path": "resnet18_best.pth",
        "classifier": ("fc", torch.nn.Linear(512, num_classes))
    },
    "densenet121": {
        "fn": densenet121,
        "path": "densenet121_best.pth",
        "classifier": ("classifier", torch.nn.Linear(1024, num_classes))
    },
    "shufflenetv2": {
        "fn": shufflenet_v2_x1_0,
        "path": "shufflenetv2_best.pth",
        "classifier": ("fc", torch.nn.Linear(1024, num_classes))
    },
    "efficientnetb0": {
        "fn": efficientnet_b0,
        "path": "efficientnetb0_best.pth",
        "classifier": ("classifier", torch.nn.Sequential(
            torch.nn.Dropout(0.2, inplace=True),
            torch.nn.Linear(1280, num_classes)
        ))
    }
}

# === Skor Hesaplayıcı ===
results = []

for name, info in model_info.items():
    model = load_model(info["fn"], info["path"], info["classifier"])
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })

# === Tablo Olarak Yazdır ===
df = pd.DataFrame(results)
print(df.to_string(index=False))
