import os
import csv
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pennylane as qml

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

N_QUBITS = 8
N_CLASSES = 10

# 32 ResNet features / 8 qubits = 4 data-upload stages
N_UPLOADS = 4

# RTX 4090 should comfortably handle this for 8 qubits.
# If you get CUDA OOM, reduce to 256.
BATCH_SIZE = 512

EPOCHS = 15

LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

NUM_WORKERS = 4

DATA_PATH = "results/cifar10_resnet32_features.npz"

RESULT_DIR = "results"
FIGURE_DIR = "figures"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


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

print(
    "PyTorch device:",
    device
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )


# ============================================================
# LOAD RESNET 32-D FEATURES
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
)


X_all = data[
    "X_train"
].astype(
    np.float32
)

y_all = data[
    "y_train"
].astype(
    np.int64
)

X_test = data[
    "X_test"
].astype(
    np.float32
)

y_test = data[
    "y_test"
].astype(
    np.int64
)

CLASS_NAMES = list(
    data["class_names"]
)


print("\n======================================")
print("DATA")
print("======================================")

print(
    "Full training:",
    X_all.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
#
# 45,000 train
# 5,000 validation
# ============================================================

indices = np.arange(
    len(X_all)
)

train_idx, val_idx = train_test_split(

    indices,

    test_size=0.10,

    random_state=SEED,

    stratify=y_all
)


X_train = X_all[
    train_idx
]

y_train = y_all[
    train_idx
]

X_val = X_all[
    val_idx
]

y_val = y_all[
    val_idx
]


print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)


# ============================================================
# TORCH DATASETS
# ============================================================

train_dataset = TensorDataset(

    torch.from_numpy(
        X_train
    ),

    torch.from_numpy(
        y_train
    )
)


val_dataset = TensorDataset(

    torch.from_numpy(
        X_val
    ),

    torch.from_numpy(
        y_val
    )
)


test_dataset = TensorDataset(

    torch.from_numpy(
        X_test
    ),

    torch.from_numpy(
        y_test
    )
)


# ============================================================
# DATA LOADERS
# ============================================================

pin_memory = (
    device.type == "cuda"
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=pin_memory
)


# ============================================================
# QUANTUM DEVICE
#
# shots=None:
# exact state-vector simulation
#
# No sampling noise yet.
# This is our IDEAL quantum baseline.
# ============================================================

qdev = qml.device(

    "default.qubit",

    wires=N_QUBITS,

    shots=None
)


# ============================================================
# BATCHED QUANTUM CIRCUIT
#
# Input:
#
# [batch_size, 32]
#
# 32 features are divided into:
#
# 0:8
# 8:16
# 16:24
# 24:32
#
# The same 8 qubits are reused four times.
# ============================================================

@qml.qnode(
    qdev,
    interface="torch",
    diff_method="backprop"
)
def quantum_circuit(
    inputs,
    weights
):

    # ========================================================
    # DATA RE-UPLOADING
    # ========================================================

    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )


        # ----------------------------------------------------
        # Encode eight features simultaneously
        #
        # inputs[:, feature]
        # has shape:
        #
        # [batch_size]
        #
        # PennyLane broadcasts the operation across
        # the complete batch.
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qml.RY(

                inputs[
                    :,
                    start + q
                ],

                wires=q
            )


        # ----------------------------------------------------
        # TRAINABLE ROTATIONS
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qml.RY(

                weights[
                    upload,
                    q,
                    0
                ],

                wires=q
            )


            qml.RZ(

                weights[
                    upload,
                    q,
                    1
                ],

                wires=q
            )


        # ----------------------------------------------------
        # ENTANGLEMENT
        #
        # Ring topology
        # ----------------------------------------------------

        for q in range(
            N_QUBITS - 1
        ):

            qml.CNOT(

                wires=[
                    q,
                    q + 1
                ]
            )


        qml.CNOT(

            wires=[
                N_QUBITS - 1,
                0
            ]
        )


    # ========================================================
    # QUANTUM OUTPUT
    #
    # 8 expectation values
    # ========================================================

    return tuple(

        qml.expval(
            qml.PauliZ(q)
        )

        for q in range(
            N_QUBITS
        )
    )


# ============================================================
# HYBRID QUANTUM CLASSIFIER
#
# 32 ResNet features
#        ↓
# 8-qubit circuit
#        ↓
# 8 expectation values
#        ↓
# classical classifier
#        ↓
# 10 CIFAR classes
# ============================================================

class QuantumClassifier(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        # ----------------------------------------------------
        # TRAINABLE QUANTUM PARAMETERS
        #
        # 4 uploads
        # × 8 qubits
        # × 2 rotations
        # ----------------------------------------------------

        self.quantum_weights = (
            nn.Parameter(

                0.05
                * torch.randn(

                    N_UPLOADS,

                    N_QUBITS,

                    2
                )
            )
        )


        # ----------------------------------------------------
        # CLASSICAL OUTPUT HEAD
        # ----------------------------------------------------

        self.classifier = (
            nn.Sequential(

                nn.Linear(
                    N_QUBITS,
                    32
                ),

                nn.ReLU(),

                nn.Dropout(
                    p=0.10
                ),

                nn.Linear(
                    32,
                    N_CLASSES
                )
            )
        )


    def forward(
        self,
        x
    ):

        # ====================================================
        # CRITICAL SPEED IMPROVEMENT
        #
        # OLD:
        #
        # for sample in x:
        #     quantum_circuit(sample)
        #
        # NEW:
        #
        # ONE quantum-circuit call
        # for the WHOLE batch.
        # ====================================================

        q_outputs = quantum_circuit(

            x,

            self.quantum_weights
        )


        # q_outputs:
        #
        # tuple length = 8
        #
        # each element:
        #
        # [batch_size]
        #
        # convert into:
        #
        # [batch_size, 8]
        # ====================================================

        quantum_features = (
            torch.stack(

                q_outputs,

                dim=1
            )
        )


        # Ensure classical network uses float32

        quantum_features = (
            quantum_features.float()
        )


        logits = self.classifier(
            quantum_features
        )


        return logits


# ============================================================
# CREATE MODEL
# ============================================================

model = QuantumClassifier()

model = model.to(
    device
)


print("\n======================================")
print("MODEL")
print("======================================")

print(
    "Model device:",
    next(
        model.parameters()
    ).device
)

print(
    "Quantum parameters:",
    model.quantum_weights.numel()
)


# ============================================================
# LOSS
# ============================================================

criterion = (
    nn.CrossEntropyLoss(
        label_smoothing=0.05
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    loader,
    return_predictions=False
):

    model.eval()


    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []


    with torch.no_grad():

        for x, y in loader:

            x = x.to(

                device,

                non_blocking=True
            )

            y = y.to(

                device,

                non_blocking=True
            )


            logits = model(
                x
            )


            loss = criterion(
                logits,
                y
            )


            total_loss += (
                loss.item()
                * y.size(0)
            )


            prediction = (
                torch.argmax(
                    logits,
                    dim=1
                )
            )


            all_predictions.extend(

                prediction
                .detach()
                .cpu()
                .numpy()
            )


            all_labels.extend(

                y
                .detach()
                .cpu()
                .numpy()
            )


            total_samples += (
                y.size(0)
            )


    average_loss = (
        total_loss
        / total_samples
    )


    accuracy = accuracy_score(

        all_labels,

        all_predictions
    )


    if return_predictions:

        return (

            average_loss,

            accuracy,

            np.array(
                all_labels
            ),

            np.array(
                all_predictions
            )
        )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

train_loss_history = []
val_loss_history = []

train_accuracy_history = []
val_accuracy_history = []


best_val_accuracy = 0.0

epochs_without_improvement = 0


BEST_MODEL_PATH = os.path.join(

    RESULT_DIR,

    "cifar10_quantum_batched_best.pt"
)


# ============================================================
# TRAINING
# ============================================================

print("\n======================================")
print("TRAINING IDEAL 8-QUBIT MODEL")
print("======================================")


total_start_time = (
    time.time()
)


for epoch in range(
    EPOCHS
):

    epoch_start_time = (
        time.time()
    )


    model.train()


    running_loss = 0.0

    running_correct = 0

    running_total = 0


    # ========================================================
    # TRAIN BATCHES
    # ========================================================

    for batch_index, (
        x,
        y
    ) in enumerate(
        train_loader
    ):


        x = x.to(

            device,

            non_blocking=True
        )


        y = y.to(

            device,

            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        logits = model(
            x
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(
            logits,
            y
        )


        # ----------------------------------------------------
        # BACKPROP
        # ----------------------------------------------------

        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        running_loss += (

            loss.item()
            * y.size(0)
        )


        prediction = (
            torch.argmax(

                logits,

                dim=1
            )
        )


        running_correct += (

            prediction
            .eq(y)
            .sum()
            .item()
        )


        running_total += (
            y.size(0)
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 20 == 0:

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

        running_loss
        / running_total
    )


    train_accuracy = (

        running_correct
        / running_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_loss,
        val_accuracy
    ) = evaluate(

        model,

        val_loader
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    train_loss_history.append(
        train_loss
    )

    val_loss_history.append(
        val_loss
    )

    train_accuracy_history.append(
        train_accuracy
    )

    val_accuracy_history.append(
        val_accuracy
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step()


    epoch_time = (
        time.time()
        - epoch_start_time
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
        f"Epoch time     = "
        f"{epoch_time:.1f} seconds"
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

        epochs_without_improvement = 0


        torch.save(

            model.state_dict(),

            BEST_MODEL_PATH
        )


        print(
            "Best model updated."
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# TOTAL TRAINING TIME
# ============================================================

total_training_time = (
    time.time()
    - total_start_time
)


print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")

print(
    f"Total training time = "
    f"{total_training_time / 60:.2f} minutes"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(

    torch.load(

        BEST_MODEL_PATH,

        map_location=device
    )
)


# ============================================================
# TEST SET
# ============================================================

(
    test_loss,
    test_accuracy,
    test_labels,
    test_predictions
) = evaluate(

    model,

    test_loader,

    return_predictions=True
)


print("\n======================================")
print("FINAL QUANTUM RESULTS")
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
# SAVE TRAINING CSV
# ============================================================

CSV_PATH = os.path.join(

    RESULT_DIR,

    "cifar10_quantum_batched_training.csv"
)


with open(
    CSV_PATH,
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
        len(
            train_loss_history
        )
    ):

        writer.writerow([

            epoch + 1,

            train_loss_history[
                epoch
            ],

            val_loss_history[
                epoch
            ],

            train_accuracy_history[
                epoch
            ],

            val_accuracy_history[
                epoch
            ]
        ])


# ============================================================
# FIGURE 1
# ACCURACY
# ============================================================

epochs = np.arange(

    1,

    len(
        train_accuracy_history
    ) + 1
)


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    epochs,

    np.array(
        train_accuracy_history
    ) * 100,

    marker="o",

    label="Train"
)


plt.plot(

    epochs,

    np.array(
        val_accuracy_history
    ) * 100,

    marker="s",

    label="Validation"
)


# Desired target
plt.axhline(

    77,

    linestyle="--",

    label="77% target"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Ideal 8-Qubit Hybrid Quantum CIFAR-10 Classifier"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_quantum_batched_accuracy.png"
)


plt.savefig(

    ACCURACY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2
# LOSS
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    epochs,

    train_loss_history,

    marker="o",

    label="Train"
)


plt.plot(

    epochs,

    val_loss_history,

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
    "Ideal Quantum Classifier Training Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


LOSS_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_quantum_batched_loss.png"
)


plt.savefig(

    LOSS_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
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

    range(
        N_CLASSES
    ),

    CLASS_NAMES,

    rotation=45,

    ha="right"
)


plt.yticks(

    range(
        N_CLASSES
    ),

    CLASS_NAMES
)


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "Ideal 8-Qubit CIFAR-10 Confusion Matrix"
)


for i in range(
    N_CLASSES
):

    for j in range(
        N_CLASSES
    ):

        plt.text(

            j,

            i,

            cm[
                i,
                j
            ],

            ha="center",

            va="center",

            fontsize=7
        )


plt.tight_layout()


CM_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_quantum_batched_confusion_matrix.png"
)


plt.savefig(

    CM_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 4
# PER-CLASS RECALL
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


plt.xlabel(
    "CIFAR-10 Class"
)

plt.ylabel(
    "Recall (%)"
)

plt.title(
    "Ideal Quantum Classifier Per-Class Recall"
)

plt.xticks(

    rotation=35,

    ha="right"
)

plt.tight_layout()


CLASS_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_quantum_batched_per_class.png"
)


plt.savefig(

    CLASS_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL FILE SUMMARY
# ============================================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "Training CSV:",
    CSV_PATH
)

print(
    "Accuracy figure:",
    ACCURACY_FIGURE
)

print(
    "Loss figure:",
    LOSS_FIGURE
)

print(
    "Confusion matrix:",
    CM_FIGURE
)

print(
    "Per-class figure:",
    CLASS_FIGURE
)