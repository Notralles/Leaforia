import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2, resnet18, densenet121, shufflenet_v2_x1_0, efficientnet_b0
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['c0', 'c1', 'c2', 'c3']
num_classes = len(class_names)
batch_size = 64

# === Ağırlıklar ===
model_weights = {
    "mobilenetv2": 0.195,
    "resnet18": 0.196,
    "densenet121": 0.202,
    "shufflenetv2": 0.188,
    "efficientnetb0": 0.219
}

# === Veri ===
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

# === Model Başlatıcıları ===
def load_model(model_fn, path, classifier_attr):
    model = model_fn(weights=None)
    setattr(model, *classifier_attr)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

models = {
    "mobilenetv2": load_model(mobilenet_v2, "mobilenetv2_best.pth", ("classifier", torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1280, num_classes)
    ))),
    "resnet18": load_model(resnet18, "resnet18_best.pth", ("fc", torch.nn.Linear(512, num_classes))),
    "densenet121": load_model(densenet121, "densenet121_best.pth", ("classifier", torch.nn.Linear(1024, num_classes))),
    "shufflenetv2": load_model(shufflenet_v2_x1_0, "shufflenetv2_best.pth", ("fc", torch.nn.Linear(1024, num_classes))),
    "efficientnetb0": load_model(efficientnet_b0, "efficientnetb0_best.pth", ("classifier", torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(1280, num_classes)
    )))
}

# === Ağırlıklı Voting ===
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        weighted_outputs = torch.zeros((inputs.size(0), num_classes), device=device)

        for name, model in models.items():
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            weighted_outputs += model_weights[name] * probs

        final_preds = torch.argmax(weighted_outputs, dim=1)
        all_preds.extend(final_preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# === Accuracy ve Confusion Matrix ===
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

acc = accuracy_score(all_labels, all_preds)
print(f"🎯 Weighted Voting Accuracy: {acc:.4f}")
print("\n🧾 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# === Confusion Matrix Plot ===
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.5)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Weighted Voting Confusion Matrix", fontsize=18)
plt.xlabel("Predicted", fontsize=16)
plt.ylabel("True", fontsize=16)
plt.tight_layout()
plt.show()
