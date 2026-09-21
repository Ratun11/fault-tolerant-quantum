import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BOTTLENECK_DIM = 32
N_CLASSES = 10

EPOCHS = 15

BATCH_SIZE = 128

BACKBONE_LR = 1e-4
HEAD_LR = 1e-3

WEIGHT_DECAY = 1e-4

DATA_DIR = "data"
RESULT_DIR = "results"
FIGURE_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n======================================")
print("DEVICE")
print("======================================")

print(device)


# ============================================================
# CIFAR-10 CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# IMAGE TRANSFORMS
#
# ResNet18 is pretrained on ImageNet.
#
# CIFAR-10:
# 32 x 32
#
# Resize to 224 x 224 for pretrained ResNet18.
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# LOAD DATASET ONCE TO GET LABELS
# ============================================================

split_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True
)

all_indices = np.arange(
    len(split_dataset)
)

all_labels = np.array(
    split_dataset.targets
)


# ============================================================
# TRAIN / VALIDATION SPLIT
#
# 45,000 training
# 5,000 validation
#
# Stratified split preserves all 10 class ratios.
# ============================================================

train_indices, val_indices = train_test_split(

    all_indices,

    test_size=0.10,

    random_state=SEED,

    stratify=all_labels
)


# ============================================================
# DATASETS WITH DIFFERENT TRANSFORMS
# ============================================================

train_full_augmented = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=train_transform
)


train_full_eval = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=eval_transform
)


test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform
)


train_dataset = Subset(
    train_full_augmented,
    train_indices
)

val_dataset = Subset(
    train_full_eval,
    val_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)


print("\n======================================")
print("DATASET")
print("======================================")

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# MODEL
#
# CIFAR-10
#      ↓
# ResNet18
#      ↓
# 512 features
#      ↓
# Linear
#      ↓
# 32-dimensional bottleneck
#      ↓
# Linear
#      ↓
# 10 CIFAR-10 classes
# ============================================================

class ResNet32Classifier(nn.Module):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # PRETRAINED RESNET18
        # ----------------------------------------------------

        weights = (
            ResNet18_Weights.DEFAULT
        )

        self.backbone = models.resnet18(
            weights=weights
        )


        backbone_features = (
            self.backbone.fc.in_features
        )


        # Remove original ImageNet classifier
        self.backbone.fc = nn.Identity()


        # ----------------------------------------------------
        # 32-D FEATURE BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = nn.Linear(
            backbone_features,
            BOTTLENECK_DIM
        )


        self.activation = nn.ReLU()


        self.dropout = nn.Dropout(
            p=0.20
        )


        # ----------------------------------------------------
        # CIFAR-10 CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            BOTTLENECK_DIM,
            N_CLASSES
        )


    def extract_features(self, x):

        x = self.backbone(x)

        x = self.bottleneck(x)

        x = self.activation(x)

        return x


    def forward(self, x):

        features = self.extract_features(
            x
        )

        features = self.dropout(
            features
        )

        logits = self.classifier(
            features
        )

        return logits


model = ResNet32Classifier()

model = model.to(
    device
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


# ============================================================
# OPTIMIZER
#
# Smaller learning rate for pretrained backbone.
# Larger learning rate for new 32-D head.
# ============================================================

optimizer = torch.optim.AdamW(

    [

        {
            "params":
                model.backbone.parameters(),

            "lr":
                BACKBONE_LR
        },

        {
            "params":
                model.bottleneck.parameters(),

            "lr":
                HEAD_LR
        },

        {
            "params":
                model.classifier.parameters(),

            "lr":
                HEAD_LR
        }

    ],

    weight_decay=WEIGHT_DECAY
)


scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0
    total = 0

    predictions_all = []
    labels_all = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels
            )


            total_loss += (
                loss.item()
                * labels.size(0)
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            predictions_all.extend(
                predictions.cpu().numpy()
            )

            labels_all.extend(
                labels.cpu().numpy()
            )


            total += labels.size(0)


    average_loss = (
        total_loss / total
    )


    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )


    return (
        average_loss,
        accuracy,
        np.array(labels_all),
        np.array(predictions_all)
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


best_val_accuracy = 0.0


best_model_path = os.path.join(
    RESULT_DIR,
    "cifar10_resnet32_best.pt"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n======================================")
print("TRAINING RESNET18 → 32D → CIFAR-10")
print("======================================")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):


        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        logits = model(
            images
        )


        loss = criterion(
            logits,
            labels
        )


        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * labels.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        total_correct += (
            predictions == labels
        ).sum().item()


        total_samples += labels.size(0)


        if (
            batch_index + 1
        ) % 100 == 0:

            print(

                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "

                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"

            )


    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_loss = (
        total_loss
        / total_samples
    )

    train_accuracy = (
        total_correct
        / total_samples
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )


    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    print(
        "\n--------------------------------------"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train loss     = "
        f"{train_loss:.4f}"
    )

    print(
        f"Validation loss= "
        f"{val_loss:.4f}"
    )

    print(
        f"Train accuracy = "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val accuracy   = "
        f"{val_accuracy:.4f}"
    )

    print(
        "--------------------------------------"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        val_accuracy
        >
        best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(
            model.state_dict(),
            best_model_path
        )


        print(
            "Best model updated."
        )


    scheduler.step()


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

(
    test_loss,
    test_accuracy,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader
)


print("\n======================================")
print("FINAL RESULTS")
print("======================================")

print(
    f"Best validation accuracy = "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test accuracy            = "
    f"{test_accuracy:.4f}"
)

print(
    f"Test loss                = "
    f"{test_loss:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    test_labels,

    test_predictions,

    target_names=CLASS_NAMES,

    output_dict=True
)


print("\n======================================")
print("PER-CLASS RECALL")
print("======================================")


per_class_recall = []


for class_name in CLASS_NAMES:

    recall = report[
        class_name
    ]["recall"]

    per_class_recall.append(
        recall
    )

    print(
        f"{class_name:12s}: "
        f"{recall:.4f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    RESULT_DIR,
    "cifar10_resnet32_training.csv"
)


with open(
    history_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ])


    for epoch in range(
        EPOCHS
    ):

        writer.writerow([

            epoch + 1,

            train_losses[epoch],

            val_losses[epoch],

            train_accuracies[epoch],

            val_accuracies[epoch]

        ])


# ============================================================
# FIGURE 1:
# TRAINING / VALIDATION ACCURACY
# ============================================================

epochs = np.arange(
    1,
    EPOCHS + 1
)


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(
    epochs,
    np.array(
        train_accuracies
    ) * 100,
    marker="o",
    label="Train"
)


plt.plot(
    epochs,
    np.array(
        val_accuracies
    ) * 100,
    marker="s",
    label="Validation"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "CIFAR-10 ResNet18 with 32-D Bottleneck"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


accuracy_path = os.path.join(
    FIGURE_DIR,
    "cifar10_resnet32_accuracy.png"
)


plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2:
# TRAINING / VALIDATION LOSS
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train"
)


plt.plot(
    epochs,
    val_losses,
    marker="s",
    label="Validation"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Cross-Entropy Loss"
)

plt.title(
    "CIFAR-10 ResNet18 Training Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


loss_path = os.path.join(
    FIGURE_DIR,
    "cifar10_resnet32_loss.png"
)


plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3:
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


plt.figure(
    figsize=(9, 8)
)


plt.imshow(
    cm
)

plt.colorbar()


plt.xticks(
    range(10),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(10),
    CLASS_NAMES
)


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "CIFAR-10 ResNet18 Confusion Matrix"
)


for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=7
        )


plt.tight_layout()


cm_path = os.path.join(
    FIGURE_DIR,
    "cifar10_resnet32_confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4:
# PER-CLASS PERFORMANCE
# ============================================================

plt.figure(
    figsize=(10, 5)
)


bars = plt.bar(
    CLASS_NAMES,
    np.array(
        per_class_recall
    ) * 100
)


for bar, value in zip(
    bars,
    per_class_recall
):

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value * 100:.1f}%",

        ha="center",
        va="bottom",
        fontsize=8

    )


plt.ylabel(
    "Recall (%)"
)

plt.xlabel(
    "CIFAR-10 Class"
)

plt.title(
    "CIFAR-10 Per-Class Recall"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()


class_path = os.path.join(
    FIGURE_DIR,
    "cifar10_resnet32_per_class.png"
)


plt.savefig(
    class_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# EXPORT 32-D FEATURES
#
# These will later become the quantum-model inputs.
#
# IMPORTANT:
# Use non-augmented evaluation images.
# ============================================================

full_train_eval_loader = DataLoader(

    train_full_eval,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=torch.cuda.is_available()
)


def extract_features(
    model,
    loader
):

    model.eval()

    all_features = []
    all_labels = []


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )


            features = (
                model.extract_features(
                    images
                )
            )


            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )


    return (
        np.concatenate(
            all_features,
            axis=0
        ),

        np.concatenate(
            all_labels,
            axis=0
        )
    )


print("\n======================================")
print("EXTRACTING 32-D FEATURES")
print("======================================")


X_train_32, y_train_32 = (
    extract_features(
        model,
        full_train_eval_loader
    )
)


X_test_32, y_test_32 = (
    extract_features(
        model,
        test_loader
    )
)


print(
    "Train features:",
    X_train_32.shape
)

print(
    "Test features:",
    X_test_32.shape
)


# ============================================================
# SCALE FEATURES TO [-pi, pi]
#
# Useful later for quantum rotations.
# ============================================================

max_abs = np.max(
    np.abs(
        X_train_32
    ),
    axis=0
)


max_abs[
    max_abs == 0
] = 1.0


X_train_quantum = (

    np.pi
    * X_train_32
    / max_abs

)


X_test_quantum = (

    np.pi
    * X_test_32
    / max_abs

)


# ============================================================
# SAVE QUANTUM-READY FEATURES
# ============================================================

feature_path = os.path.join(
    RESULT_DIR,
    "cifar10_resnet32_features.npz"
)


np.savez_compressed(

    feature_path,

    X_train=X_train_quantum,

    y_train=y_train_32,

    X_test=X_test_quantum,

    y_test=y_test_32,

    raw_train_features=X_train_32,

    raw_test_features=X_test_32,

    max_abs=max_abs,

    class_names=np.array(
        CLASS_NAMES
    )
)


print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Best model:",
    best_model_path
)

print(
    "Training history:",
    history_path
)

print(
    "Quantum-ready features:",
    feature_path
)

print(
    "Accuracy figure:",
    accuracy_path
)

print(
    "Loss figure:",
    loss_path
)

print(
    "Confusion matrix:",
    cm_path
)

print(
    "Per-class figure:",
    class_path
)