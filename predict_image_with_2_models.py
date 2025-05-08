import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import squeezenet1_0, shufflenet_v2_x1_0
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy'
]
num_classes = len(class_names)

# Test verisini yükle
test_dir = "C:/dataset/test"  # Burayı test klasörünün tam yoluna göre değiştir
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modelleri yükle
def load_squeezenet(num_classes):
    model = squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    model.load_state_dict(torch.load("squeezenet.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("shufflenet.pth", map_location=device))
    model.eval()
    return model.to(device)

# Değerlendirme fonksiyonu
def evaluate_model(model, dataloader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(model, squeezenet1_0().__class__):
                outputs = outputs.squeeze()
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

# Modelleri değerlendir
print("------ SqueezeNet Sonuçları ------")
squeezenet_model = load_squeezenet(num_classes)
evaluate_model(squeezenet_model, test_loader)

print("\n------ ShuffleNet Sonuçları ------")
shufflenet_model = load_shufflenet(num_classes)
evaluate_model(shufflenet_model, test_loader)
