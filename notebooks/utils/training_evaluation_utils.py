# Import the relevant libraries
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, auc)
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

ACC_WEIGHT = 0.5
MACRO_F1_WEIGHT = 0.5

# Function for training loop for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    # Set the model to training mode
    model.train()
    # Initialise
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    # Iterate through each training batch
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        outputs = model(inputs)

        # Special handling for the Inception-V3 model
        if hasattr(outputs, "logits") and hasattr(outputs, "aux_logits"):
            main_outputs = outputs.logits
            auxiliary_outputs = outputs.aux_logits

            loss1 = criterion(main_outputs, labels)
            loss2 = criterion(auxiliary_outputs, labels)
            loss = loss1 + 0.4 * loss2
            outputs = main_outputs

        # For other models like ResNet, VGG, DenseNet, EfficientNet
        else:
            loss = criterion(outputs, labels)

        # Convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        # Backpropagation
        loss.backward()
        # Clip gratdient (prevent large gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        # Update model parameters
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Save prediction and labels
        all_predictions.extend(predicted_class.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    epoch_f1 = calculate_macro_f1(all_labels, all_predictions)

    return epoch_loss, epoch_acc, epoch_f1

# Function to train the model
def train_model(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler, device, num_epochs, patience,stage_name
):
    # Store copy of best model weights so far
    best_model_weights = copy.deepcopy(model.state_dict())
    best_score = 0.0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_score": []
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n[{stage_name}] Epoch {epoch + 1}/{num_epochs}")
        # Train one epoch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        # Evaluate model on validation set
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        val_score = calculate_combined_score(val_acc, val_f1)

        if scheduler is not None:
            scheduler.step(val_score)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_score"].append(val_score)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f} | "
            f"Val Score: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"New best score: {best_score:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    elapsed = time.time() - start_time

    print(f"\n{stage_name} finished in {elapsed / 60:.2f} minutes")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Best Combined Score: {best_score:.4f}")

    model.load_state_dict(best_model_weights)

    return model, history, best_score, best_val_acc, best_val_f1


# Function to calculate macro f1-score
def calculate_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

# Function to calculate combined score
def calculate_combined_score(acc, macro_f1):
    return (ACC_WEIGHT * acc) + (MACRO_F1_WEIGHT * macro_f1)

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        # InceptionV3
        if hasattr(outputs, "logits"):
            main_outputs = outputs.logits

            if outputs.aux_logits is not None:
                aux_outputs = outputs.aux_logits

                loss1 = criterion(main_outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                loss = criterion(main_outputs, labels)

            outputs = main_outputs

        # Models that return tuple
        elif isinstance(outputs, tuple):
            main_outputs, aux_outputs = outputs
            loss1 = criterion(main_outputs, labels)
            loss2 = criterion(aux_outputs, labels)

            loss = loss1 + 0.4 * loss2
            outputs = main_outputs

        # ResNet / DenseNet / EfficientNet / VGG
        else:
            loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * inputs.size(0)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = calculate_macro_f1(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds, np.array(all_probs)

# Function to evaluate the model with TTA
@torch.no_grad()
def evaluate_tta(model, loader, criterion, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    running_loss = 0.0

    for tta_inputs, labels in loader:
        labels = labels.to(device)

        # Shape: [1, num_tta, C, H, W] -> [num_tta, C, H, W]
        tta_inputs = tta_inputs.squeeze(0).to(device)

        outputs = model(tta_inputs)

         # Handles InceptionV3 output
        if hasattr(outputs, "logits"):
            outputs = outputs.logits


        # Average logits for loss
        avg_logits = outputs.mean(dim=0, keepdim=True)

        # Average probabilities for prediction and ROC-AUC
        probs_each = torch.softmax(outputs, dim=1)
        avg_probs = probs_each.mean(dim=0, keepdim=True)

        pred = torch.argmax(avg_probs, dim=1)

        loss = criterion(avg_logits, labels)

        running_loss += loss.item()

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(pred.detach().cpu().numpy())
        all_probs.extend(avg_probs.detach().cpu().numpy())

    tta_loss = running_loss / len(loader.dataset)
    tta_acc = accuracy_score(all_labels, all_preds)
    tta_f1 = f1_score(all_labels, all_preds, average="macro")

    return tta_loss, tta_acc, tta_f1, all_labels, all_preds, np.array(all_probs)


# Function to plot the training curves
def plot_history(hist1, hist2):
    train_loss = hist1["train_loss"] + hist2["train_loss"]
    val_loss = hist1["val_loss"] + hist2["val_loss"]
    train_f1 = hist1["train_f1"] + hist2["train_f1"]
    val_f1 = hist1["val_f1"] + hist2["val_f1"]
    train_acc = hist1["train_acc"] + hist2["train_acc"]
    val_acc = hist1["val_acc"] + hist2["val_acc"]
    val_score = hist1["val_score"] + hist2["val_score"]

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet-34 Loss Curve")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet-34 Accuracy Curve")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1, label="Train Macro F1")
    plt.plot(epochs, val_f1, label="Val Macro F1")
    plt.plot(epochs, val_score, label="Val Combined Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("ResNet-34 Macro F1 / Combined Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(
    y_true,
    y_prob,
    class_names,
    title="Multi-class ROC Curve"
):
    num_classes = len(class_names)

    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(num_classes)
    )

    macro_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    weighted_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="weighted",
        multi_class="ovr"
    )

    roc_auc_per_class = {}

    print("\n=================== ROC-AUC ===================")
    print(f"Macro ROC-AUC    : {macro_auc:.4f}")
    print(f"Weighted ROC-AUC : {weighted_auc:.4f}")

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        class_auc = auc(fpr, tpr)
        roc_auc_per_class[class_names[i]] = float(class_auc)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {class_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return macro_auc, weighted_auc, roc_auc_per_class



@torch.no_grad()
def tta_predict(model, images):
    # images shape: [batch, C, H, W]

    tta_images = [
        images,
        # Horizontal flip
        torch.flip(images, dims=[3]),
        # Vertical flip
        torch.flip(images, dims=[2]),
        # Horizontal and Vertical flip
        torch.flip(images, dims=[2, 3]),
        TF.rotate(images, angle=10),
        TF.rotate(images, angle=-10)
    ]

    probs = []

    for img in tta_images:
        outputs = model(img)
        probs.append(torch.softmax(outputs, dim=1))

    avg_probs = torch.stack(probs).mean(dim=0)

    return avg_probs


# Perform McNemar's test comparing normal inference and TTA
def calculate_mcnemar(y_true, y_pred, y_true_tta, y_pred_tta):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_tta = np.array(y_true_tta)
    y_pred_tta = np.array(y_pred_tta)

    assert np.array_equal( y_true, y_true_tta), "The order of samples differs between normal inference and TTA."

    without_correct = (y_pred == y_true)
    with_correct = (y_pred_tta == y_true)

    A = np.sum(without_correct & with_correct)
    B = np.sum(without_correct & ~with_correct)
    C = np.sum(~without_correct & with_correct)
    D = np.sum(~without_correct & ~with_correct)

    table = [
        [int(A), int(B)],
        [int(C), int(D)]
    ]

    result = mcnemar(table, exact=True)

    return {
        "contingency_table": table,
        "correct_only_without_tta": int(B),
        "correct_only_with_tta": int(C),
        "p_value": float(result.pvalue)
    }