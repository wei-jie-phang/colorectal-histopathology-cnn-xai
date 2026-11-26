# Import the relevant libraries
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from utils.data_utils import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

DATA_DIR = "../data/EBHI-SEG"
classes, class_to_index, samples = collect_samples(DATA_DIR)
class_names = classes

# Class for the External Dataset (LC25000)
class ExternalDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# Class for the External Dataset (LC25000) for Test-Time Augmentation
class ExternalTTADataset(Dataset):
    def __init__(self, samples, image_size=224):
        self.samples = samples
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def preprocess(self, img):
        img = TF.resize(img, [self.image_size, self.image_size])
        img = TF.to_tensor(img)
        img = self.normalize(img)
        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, [self.image_size, self.image_size])

        variants = [
            img,
            TF.hflip(img),
            TF.vflip(img),
            TF.hflip(TF.vflip(img)),
            TF.rotate(img, 10),
            TF.rotate(img, -10)
        ]

        tensors = [self.preprocess(v) for v in variants]

        return torch.stack(tensors), label


# Preprocessing for evaluation set
def get_eval_transform(image_size):
    # (1)Resize, (2) Convert to PyTorch tensor, (3) Normalise image
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Function to collect the external samples
def collect_external_samples(external_directory, external_label_map, class_to_index):
    samples = []

    # Loop through each folder
    for folder, ebhi_class in external_label_map.items():
        folder_path = os.path.join(external_directory, folder)

        # Check if folder exists
        if not os.path.isdir(folder_path):
            print(f"Missing folder: {folder_path}")
            continue
        # Convert class name to numerical label
        label_index = class_to_index[ebhi_class]

        for file in sorted(os.listdir(folder_path)):
            # Keep only image file
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                samples.append((os.path.join(folder_path, file), label_index))

    if len(samples) == 0:
        raise ValueError("No external validation images found. Check EXTERNAL_DIR.")

    return samples

# Function to create the data loader for the external dataset
def create_external_loader( external_directory, external_label_map, class_to_index, image_size, batch_size, num_workers=0):
    samples = collect_external_samples(external_directory, external_label_map, class_to_index)
    transform = get_eval_transform(image_size)

    dataset = ExternalDataset(samples, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader, samples

# Function to create the data loader for the external dataset with TTA
def create_external_tta_loader(samples, image_size, num_workers=0):
    dataset = ExternalTTADataset(samples, image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader

# Function to evaluate trained CNN on LC25000 external dataset
@torch.no_grad()
def evaluate_external(model, loader, criterion, device):
    model.eval()
    # Initialise
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    # Loop through each batch
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        running_loss += loss.item() * inputs.size(0)
        # Save
        all_predictions.extend(predicted_class.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    return avg_loss, all_labels, all_predictions, np.array(all_probabilities)

# Function to evaluate trained CNN on LC25000 external dataset with TTA
@torch.no_grad()
def evaluate_external_tta(model, loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for variants, label in loader:
        variants = variants.squeeze(0).to(device)

        outputs = model(variants)
        probabilities = torch.softmax(outputs, dim=1)

        avg_probabilities = probabilities.mean(dim=0)
        pred = torch.argmax(avg_probabilities).item()

        all_labels.append(label.item())
        all_predictions.append(pred)
        all_probabilities.append(avg_probabilities.cpu().numpy())

    return all_labels, all_predictions, np.array(all_probabilities)

# Function to calculate the external validation metrics
def calculate_external_metrics(y_true, y_pred, external_labels, external_target_names):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=external_labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=external_labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=external_labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, labels=external_labels, target_names=external_target_names,
            digits=4, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=external_labels)
    }

    return results

def calculate_external_metrics(y_true, y_pred, external_labels, external_target_names, class_names):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    all_labels = list(range(len(class_names)))

    # Standard metrics for the two mapped external classes
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred,
            labels=external_labels,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred,
            labels=external_labels,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=external_labels,
            average="macro",
            zero_division=0
        ),
        "classification_report_external_only": classification_report(
            y_true, y_pred,
            labels=external_labels,
            target_names=external_target_names,
            digits=4,
            zero_division=0
        ),
        # Current 2x2 metrics
        "confusion_matrix_external_only": confusion_matrix(
            y_true, y_pred,
            labels=external_labels
        ),
        # Full 6x6 matrix.
        # This shows all predicted classes.
        "confusion_matrix_full_6_class": confusion_matrix(
            y_true, y_pred,
            labels=all_labels
        )
    }

    # Build 2x3 matrix: True external class vs Predicted ACA / Normal / Other
    rows = []
    for true_label in external_labels:
        true_mask = y_true == true_label

        row = []
        for pred_label in external_labels:
            row.append(int(np.sum(true_mask & (y_pred == pred_label))))

        other_count = int(np.sum(true_mask & ~np.isin(y_pred, external_labels)))
        row.append(other_count)

        rows.append(row)

    results["confusion_matrix_external_with_other"] = np.array(rows)
    results["confusion_matrix_external_with_other_columns"] = (
        external_target_names + ["Other EBHI-SEG classes"])

    results["confusion_matrix_external_with_other_rows"] = external_target_names

    return results

# Function to print the external validation results
def print_external_results(title, loss, results):
    print(f"\n=================== {title} ===================")

    if loss is not None:
        print(f"External Loss      : {loss:.4f}")

    print(f"External Accuracy  : {results['accuracy']:.4f}")
    print(f"External Precision : {results['precision']:.4f}")
    print(f"External Recall    : {results['recall']:.4f}")
    print(f"External Macro F1  : {results['macro_f1']:.4f}")

    print("\nClassification Report:")
    print(results["classification_report_external_only"])

    print("\n2x2 Confusion Matrix: External Classes Only")
    print(results["confusion_matrix_external_only"])

    print("\n2x3 Confusion Matrix: External Classes + Other EBHI-SEG Predictions")
    print("Rows   :", results["confusion_matrix_external_with_other_rows"])
    print("Columns:", results["confusion_matrix_external_with_other_columns"])
    print(results["confusion_matrix_external_with_other"])

    print("\nFull 6-Class Confusion Matrix:")
    print("Rows/Columns:", class_names)
    print(results["confusion_matrix_full_6_class"])
