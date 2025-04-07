import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

from torchvision.models import (

    mobilenet_v3_small, MobileNet_V3_Small_Weights,

    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,

    squeezenet1_1, SqueezeNet1_1_Weights

)

import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

import matplotlib.pyplot as plt

import seaborn as sns



# Device selection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")



# Data transformations

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(224),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



test_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



# Dataset paths

train_dir = "path/train"

test_dir = "path/test"



# Dataset and DataLoaders

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)



num_classes = len(train_dataset.classes)

print(f"Total number of classes: {num_classes}")

print(f"Classes: {train_dataset.classes}")



batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



# Model preparation functions

def prepare_mobilenet_v3(num_classes):

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model



def prepare_shufflenet_v2(num_classes):

    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model



def prepare_squeezenet(num_classes):

    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

    model.num_classes = num_classes

    return model



# Training function

def train_model(model, dataloader, criterion, optimizer, epochs=10):

    model.to(device)

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model



# Evaluation function

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



# Ensemble predictions

def ensemble_predictions(model_probs_list, weights=None):

    if weights is None:

        weights = [1/len(model_probs_list)] * len(model_probs_list)

    ensemble_probs = np.zeros_like(model_probs_list[0])

    for i, probs in enumerate(model_probs_list):

        ensemble_probs += weights[i] * probs

    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    return ensemble_preds, ensemble_probs



# Confusion matrix visualization

def plot_confusion_matrix(cm, class_names, title):

    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted')

    plt.ylabel('Actual')

    plt.title(title)

    plt.tight_layout()

    plt.show()



# Initialize models

mobilenet_model = prepare_mobilenet_v3(num_classes)

shufflenet_model = prepare_shufflenet_v2(num_classes)

squeezenet_model = prepare_squeezenet(num_classes)



# Loss function and optimizers

criterion = nn.CrossEntropyLoss()

mobilenet_optimizer = optim.Adam(mobilenet_model.parameters(), lr=0.0001)

shufflenet_optimizer = optim.Adam(shufflenet_model.parameters(), lr=0.0001)

squeezenet_optimizer = optim.Adam(squeezenet_model.parameters(), lr=0.0001)



# Training

print("Training MobileNetV3 Small...")

mobilenet_model = train_model(mobilenet_model, train_loader, criterion, mobilenet_optimizer)



print("\nTraining ShuffleNetV2...")

shufflenet_model = train_model(shufflenet_model, train_loader, criterion, shufflenet_optimizer)



print("\nTraining SqueezeNet...")

squeezenet_model = train_model(squeezenet_model, train_loader, criterion, squeezenet_optimizer)



# Evaluation

print("\nEvaluating models...")

y_true, mobilenet_preds, mobilenet_probs = evaluate_model(mobilenet_model, test_loader)

_, shufflenet_preds, shufflenet_probs = evaluate_model(shufflenet_model, test_loader)

_, squeezenet_preds, squeezenet_probs = evaluate_model(squeezenet_model, test_loader)



# Ensemble

model_probs = [mobilenet_probs, shufflenet_probs, squeezenet_probs]

ensemble_preds, ensemble_probs = ensemble_predictions(model_probs)



# Metric computation

def compute_metrics(y_true, y_pred):

    return {

        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(y_true, y_pred, average='weighted'),

        "recall": recall_score(y_true, y_pred, average='weighted'),

        "f1": f1_score(y_true, y_pred, average='weighted'),

        "confusion_matrix": confusion_matrix(y_true, y_pred)

    }



mobilenet_metrics = compute_metrics(y_true, mobilenet_preds)

shufflenet_metrics = compute_metrics(y_true, shufflenet_preds)

squeezenet_metrics = compute_metrics(y_true, squeezenet_preds)

ensemble_metrics = compute_metrics(y_true, ensemble_preds)



# Print metrics

def print_metrics(name, metrics):

    print(f"\n{name} Performance Metrics:")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    print(f"Precision: {metrics['precision']:.4f}")

    print(f"Recall:    {metrics['recall']:.4f}")

    print(f"F1 Score:  {metrics['f1']:.4f}")



print_metrics("MobileNetV3 Small", mobilenet_metrics)

print_metrics("ShuffleNetV2", shufflenet_metrics)

print_metrics("SqueezeNet", squeezenet_metrics)

print_metrics("Ensemble", ensemble_metrics)



# Confusion matrices

plot_confusion_matrix(mobilenet_metrics["confusion_matrix"], train_dataset.classes, "MobileNetV3 Confusion Matrix")

plot_confusion_matrix(shufflenet_metrics["confusion_matrix"], train_dataset.classes, "ShuffleNetV2 Confusion Matrix")

plot_confusion_matrix(squeezenet_metrics["confusion_matrix"], train_dataset.classes, "SqueezeNet Confusion Matrix")

plot_confusion_matrix(ensemble_metrics["confusion_matrix"], train_dataset.classes, "Ensemble Model Confusion Matrix")



# Save models

torch.save(mobilenet_model.state_dict(), "mobilenetv3_small_model.pth")

torch.save(shufflenet_model.state_dict(), "shufflenetv2_model.pth")

torch.save(squeezenet_model.state_dict(), "squeezenet_model.pth")

print("\nModels saved successfully.")

