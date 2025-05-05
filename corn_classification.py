import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import squeezenet1_0, mobilenet_v2, efficientnet_b0, densenet121, shufflenet_v2_x1_0
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm

# -------------------- Ayarlar --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4
batch_size = 32
num_epochs = 30
patience = 10
dataset_path = 'C:/dataset'
save_path = './saved_models'
os.makedirs(save_path, exist_ok=True)

# -------------------- Class İsimleri --------------------
class_names = ['c0', 'c1', 'c2', 'c3']

# -------------------- Veriyi Hazırlama --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def prepare_dataloaders():
    test_full = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=transform)
    val_size = int(0.1 * len(test_full))
    test_size = len(test_full) - val_size
    test_dataset, val_dataset = random_split(test_full, [test_size, val_size])

    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# -------------------- Model Başlatıcıları --------------------
def init_squeezenet(num_classes):
    model = squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    return model.to(device)

def init_mobilenetv2(num_classes):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)

def init_efficientnetb0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def init_densenet121(num_classes):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(device)

def init_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# -------------------- Early Stopping --------------------
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# -------------------- Eğitim Fonksiyonu --------------------
def train_model(model, model_name, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopper = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []
    best_model = model
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}.pth"))

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    plot_losses(train_losses, val_losses, model_name)
    return best_model

# -------------------- Loss Grafiği --------------------
def plot_losses(train_losses, val_losses, name):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{name}_loss.png')
    plt.close()

# -------------------- Değerlendirme --------------------
def evaluate(model, test_loader, name):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            output = model(images)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    print(f"\n{name} Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Reds', colorbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for texts in disp.text_.ravel():
        texts.set_fontsize(14)
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()
    return y_probs, y_pred

# -------------------- Ensemble Yöntemleri --------------------
def ensemble_predictions(models, test_loader, method='soft'):
    all_probs = []
    all_preds = []
    for model in models:
        model.eval()
        probs = []
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                output = model(images)
                prob = torch.softmax(output, dim=1)
                probs.append(prob.cpu().numpy())
        all_probs.append(np.concatenate(probs))

    all_probs = np.stack(all_probs)  # shape: [models, samples, classes]

    if method == 'soft':
        avg_probs = np.mean(all_probs, axis=0)
        final_preds = np.argmax(avg_probs, axis=1)

    elif method == 'hard':
        preds = np.argmax(all_probs, axis=2)  # shape: [models, samples]
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

    elif method == 'consensus':
        preds = np.argmax(all_probs, axis=2)
        final_preds = []
        for sample_preds in preds.T:
            counts = np.bincount(sample_preds)
            if np.max(counts) == len(models):
                final_preds.append(np.argmax(counts))
            else:
                final_preds.append(random.choice(sample_preds))
        final_preds = np.array(final_preds)

    return final_preds

def evaluate_ensemble(final_preds, test_loader, name):
    y_true = []
    for _, labels in test_loader:
        y_true.extend(labels.numpy())
    print(f"\n{name} Classification Report:\n", classification_report(y_true, final_preds, target_names=class_names))
    cm = confusion_matrix(y_true, final_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Reds', colorbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for texts in disp.text_.ravel():
        texts.set_fontsize(14)
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()

# -------------------- Ana Çalışma --------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_dataloaders()

    models_info = {
        "squeezenet": init_squeezenet,
        "mobilenetv2": init_mobilenetv2,
        "efficientnetb0": init_efficientnetb0,
        "densenet121": init_densenet121,
        "shufflenet": init_shufflenet
    }

    trained_models = []
    for name, init_func in models_info.items():
        model = init_func(num_classes)
        model = train_model(model, name, train_loader, val_loader)
        _, _ = evaluate(model, test_loader, name)
        trained_models.append(model)

    # Ensemble
    for method in ['soft', 'hard', 'consensus']:
        preds = ensemble_predictions(trained_models, test_loader, method=method)
        evaluate_ensemble(preds, test_loader, f'ensemble_{method}')
