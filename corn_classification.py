import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small, mobilenet_v2, efficientnet_b0, squeezenet1_0, shufflenet_v2_x1_0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = r"C:\dataset"
test_dir = os.path.join(dataset_dir, "test")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Model loading functions
def load_mobilenet_v3(num_classes):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv3_model.pth"))
    return model.to(device).eval()

def load_mobilenet_v2(num_classes):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv2_model.pth"))
    return model.to(device).eval()

def load_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("efficientnetb0_model.pth"))
    return model.to(device).eval()

def load_squeezenet(num_classes):
    model = squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load("squeezenet1.0_model.pth"))
    return model.to(device).eval()

def load_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("shufflenetv2_model.pth"))
    return model.to(device).eval()

# Ensemble predictions
def ensemble_predictions(model_probs_list, weights=None):
    if weights is None:
        weights = [1/len(model_probs_list)] * len(model_probs_list)
    ensemble_probs = np.zeros_like(model_probs_list[0])
    for i, probs in enumerate(model_probs_list):
        ensemble_probs += weights[i] * probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, ensemble_probs

# Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Compute metrics
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

# Print metrics
def print_metrics(name, metrics):
    print(f"\n{name} Performance Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

# Plot confusion matrix with c0, c1, c2, c3
def plot_confusion_matrix(cm, title):
    # Class names as c0, c1, c2, c3
    class_names = ['c0', 'c1', 'c2', 'c3']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_classes = 4  # Since we have 4 classes

    # Load models
    mobilenetv3_model = load_mobilenet_v3(num_classes)
    mobilenetv2_model = load_mobilenet_v2(num_classes)
    efficientnetb0_model = load_efficientnet_b0(num_classes)
    squeezenet_model = load_squeezenet(num_classes)
    shufflenet_model = load_shufflenet(num_classes)

    print("\nEvaluating models...")
    y_true, mobv3_preds, mobv3_probs = evaluate_model(mobilenetv3_model, test_loader)
    _, mobv2_preds, mobv2_probs = evaluate_model(mobilenetv2_model, test_loader)
    _, eff_preds, eff_probs = evaluate_model(efficientnetb0_model, test_loader)
    _, squeeze_preds, squeeze_probs = evaluate_model(squeezenet_model, test_loader)
    _, shuffle_preds, shuffle_probs = evaluate_model(shufflenet_model, test_loader)

    # Ensemble the models
    all_probs = [mobv3_probs, mobv2_probs, eff_probs, squeeze_probs, shuffle_probs]
    ensemble_preds, ensemble_probs = ensemble_predictions(all_probs)

    # Compute and print metrics
    metrics = {
        "MobileNetV3": compute_metrics(y_true, mobv3_preds),
        "MobileNetV2": compute_metrics(y_true, mobv2_preds),
        "EfficientNetB0": compute_metrics(y_true, eff_preds),
        "SqueezeNet": compute_metrics(y_true, squeeze_preds),
        "ShuffleNet": compute_metrics(y_true, shuffle_preds),
        "Ensemble": compute_metrics(y_true, ensemble_preds)
    }

    for name, met in metrics.items():
        print_metrics(name, met)
        plot_confusion_matrix(met["confusion_matrix"], f"{name} Confusion Matrix")
