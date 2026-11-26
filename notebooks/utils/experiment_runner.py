# This is the shared experiment pipeline that is used by all six CNN notebooks.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from utils.seed_utils import set_seed
from utils.data_utils import (collect_samples, stratified_split, get_train_transform, get_eval_transform,
                              EBHIClassificationDataset, EBHITTADataset, create_dataloaders,)
from utils.training_evaluation_utils import (train_model, evaluate, 
                                             evaluate_tta, plot_multiclass_roc, calculate_mcnemar,)
from utils.external_validation_utils import (create_external_loader, calculate_external_metrics,
                                             create_external_tta_loader, evaluate_external, evaluate_external_tta,)

def run_single_experiment(
    run_idx, seed, *, model_name, build_model, freeze_backbone,
    unfreeze_for_finetuning, data_dir, split_seed, image_size,
    batch_size, num_workers, device, label_smoothing, lr_stage1,
    lr_stage2, weight_decay, num_epochs_stage1, num_epochs_stage2,
    patience, external_dir, external_label_map, external_class_to_index,
    external_image_size, external_batch_size, external_num_workers,
    external_labels, external_target_names, external_class_names,
    save_model_per_run, model_path_fn,):
    print("\n" + "=" * 70)
    print(f"RUN {run_idx}  (seed = {seed})")
    print("=" * 70)

    set_seed(seed)

    # Preprocessing
    classes, class_to_index, samples = collect_samples(data_dir)
    class_names = classes
    num_classes = len(class_names)

    if len(samples) == 0:
        raise ValueError("No images found. Check your folder structure.")

    train_samples, val_samples, test_samples = stratified_split(
        samples,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=split_seed
    )

    train_transform = get_train_transform(image_size)
    eval_transform = get_eval_transform(image_size)

    train_dataset = EBHIClassificationDataset(train_samples, transform=train_transform)
    val_dataset = EBHIClassificationDataset(val_samples, transform=eval_transform)
    test_dataset = EBHIClassificationDataset(test_samples, transform=eval_transform)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )

    train_targets = np.array([s[1] for s in train_dataset.samples])
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_targets),
        y=train_targets
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(device)

    # Build the model
    model = build_model(num_classes).to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # Stage 1: Frozen Head Training
    optimizer_stage1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=weight_decay
    )
    scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage1, mode="max", factor=0.5, patience=2
    )

    model, hist_stage1, best_score_stage1, best_acc_stage1, best_f1_stage1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        device=device,
        num_epochs=num_epochs_stage1,
        patience=patience,
        stage_name=f"[Run {run_idx}] Frozen Head Training"
    )

    # Stage 2: Fine-Tuning
    unfreeze_for_finetuning(model)

    optimizer_stage2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage2,
        weight_decay=weight_decay
    )
    scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage2, mode="max", factor=0.5, patience=2
    )

    model, hist_stage2, best_score_stage2, best_acc_stage2, best_f1_stage2 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        device=device,
        num_epochs=num_epochs_stage2,
        patience=patience,
        stage_name=f"[Run {run_idx}] Fine-Tuning"
    )

    # Internal Test Evaluation (Normal)
    test_loss, test_acc, test_f1, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, device
    )
    test_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    test_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    macro_auc, weighted_auc, roc_auc_per_class = plot_multiclass_roc(
        y_true, y_prob, class_names,
        title=f"[Run {run_idx}] {model_name} Normal Test ROC Curve"
    )
    plt.close("all")

    # Internal Test Evaluation (TTA)
    tta_dataset = EBHITTADataset(test_samples, image_size=image_size)
    tta_loader = DataLoader(
        tta_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    tta_loss, tta_acc, tta_f1, y_true_tta, y_pred_tta, y_prob_tta = evaluate_tta(
        model, tta_loader, criterion, device
    )
    tta_precision = precision_score(y_true_tta, y_pred_tta, average="macro", zero_division=0)
    tta_recall = recall_score(y_true_tta, y_pred_tta, average="macro", zero_division=0)
    cm_tta = confusion_matrix(y_true_tta, y_pred_tta)
    macro_auc_tta, weighted_auc_tta, roc_auc_per_class_tta = plot_multiclass_roc(
        y_true_tta, y_prob_tta, class_names,
        title=f"[Run {run_idx}] {model_name} TTA Test ROC Curve"
    )
    plt.close("all")

    mcnemar_results = calculate_mcnemar(y_true, y_pred, y_true_tta, y_pred_tta)

    # External Validation (Normal)
    external_loader, external_samples = create_external_loader(
        external_dir,
        external_label_map,
        external_class_to_index,
        external_image_size,
        external_batch_size,
        external_num_workers
    )
    external_criterion = nn.CrossEntropyLoss()

    external_loss, y_true_ext, y_pred_ext, y_prob_ext = evaluate_external(
        model, external_loader, external_criterion, device
    )
    ext_results = calculate_external_metrics(
        y_true_ext, y_pred_ext, external_labels, external_target_names, external_class_names
    )

    # External Validation (TTA)
    external_tta_loader = create_external_tta_loader(
        external_samples, external_image_size, external_num_workers
    )
    y_true_ext_tta, y_pred_ext_tta, y_prob_ext_tta = evaluate_external_tta(
        model, external_tta_loader, device
    )
    ext_tta_results = calculate_external_metrics(
        y_true_ext_tta, y_pred_ext_tta, external_labels, external_target_names, external_class_names
    )

    external_mcnemar_results = calculate_mcnemar(
        y_true_ext, y_pred_ext, y_true_ext_tta, y_pred_ext_tta
    )

    if save_model_per_run:
        torch.save(model.state_dict(), model_path_fn(run_idx, seed))

    # Assemble all the results for this run 
    result = {
        "run_idx": run_idx,
        "seed": seed,
        "class_names": class_names,

        "val_stage1": {
            "best_score": float(best_score_stage1),
            "best_accuracy": float(best_acc_stage1),
            "best_macro_f1": float(best_f1_stage1),
        },
        "val_stage2": {
            "best_score": float(best_score_stage2),
            "best_accuracy": float(best_acc_stage2),
            "best_macro_f1": float(best_f1_stage2),
        },

        "internal_normal": {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "precision_macro": float(test_precision),
            "recall_macro": float(test_recall),
            "f1_macro": float(test_f1),
            "macro_roc_auc": float(macro_auc),
            "weighted_roc_auc": float(weighted_auc),
        },
        "internal_normal_extra": {
            "per_class_roc_auc": roc_auc_per_class,
            "confusion_matrix": cm.tolist(),
        },

        "internal_tta": {
            "loss": float(tta_loss),
            "accuracy": float(tta_acc),
            "precision_macro": float(tta_precision),
            "recall_macro": float(tta_recall),
            "f1_macro": float(tta_f1),
            "macro_roc_auc": float(macro_auc_tta),
            "weighted_roc_auc": float(weighted_auc_tta),
        },
        "internal_tta_extra": {
            "per_class_roc_auc": roc_auc_per_class_tta,
            "confusion_matrix": cm_tta.tolist(),
        },

        "internal_mcnemar": mcnemar_results,

        "external_normal": ext_results,
        "external_tta": ext_tta_results,
        "external_mcnemar": external_mcnemar_results,
    }
    return result
