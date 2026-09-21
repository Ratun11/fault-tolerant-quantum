import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pennylane as qml

from sklearn.metrics import accuracy_score


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

N_QUBITS = 8
N_CLASSES = 10

# 32 features / 8 qubits = 4 re-upload blocks
N_UPLOADS = 4

BATCH_SIZE = 128
EPOCHS = 15

LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4

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


# ============================================================
# DEVICE
#
# Classical calculations can use GPU.
# PennyLane default.qubit simulation remains separate.
# ============================================================

torch_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n======================================")
print("DEVICE")
print("======================================")

print("PyTorch:", torch_device)


# ============================================================
# LOAD RESNET FEATURES
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X_train = data["X_train"].astype(
    np.float32
)

y_train = data["y_train"].astype(
    np.int64
)

X_test = data["X_test"].astype(
    np.float32
)

y_test = data["y_test"].astype(
    np.int64
)

class_names = data["class_names"]


print("\n======================================")
print("DATA")
print("======================================")

print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

N_TRAIN = 45000

X_val = X_train[N_TRAIN:]
y_val = y_train[N_TRAIN:]

X_train = X_train[:N_TRAIN]
y_train = y_train[:N_TRAIN]


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

val_dataset = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_val)
)

test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# QUANTUM DEVICE
# ============================================================

qdev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# QUANTUM CIRCUIT
#
# 32 features are uploaded as:
#
# features 0:8
# features 8:16
# features 16:24
# features 24:32
#
# onto the SAME 8 qubits.
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

    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )

        # ----------------------------------------------------
        # DATA RE-UPLOADING
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            qml.RY(
                inputs[
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
        # ENTANGLING RING
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
    # QUANTUM FEATURES
    # ========================================================

    return [
        qml.expval(
            qml.PauliZ(q)
        )
        for q in range(
            N_QUBITS
        )
    ]


# ============================================================
# HYBRID QUANTUM CLASSIFIER
# ============================================================

class QuantumHead(nn.Module):

    def __init__(self):

        super().__init__()


        self.quantum_weights = nn.Parameter(

            0.05
            * torch.randn(
                N_UPLOADS,
                N_QUBITS,
                2
            )

        )


        # ----------------------------------------------------
        # Quantum output = 8 features
        #
        # Small classical classifier:
        # 8 -> 32 -> 10
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                N_QUBITS,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                32,
                N_CLASSES
            )

        )


    def forward(self, x):

        quantum_features = []


        for sample in x:

            q = quantum_circuit(
                sample,
                self.quantum_weights
            )


            q = torch.stack(
                list(q)
            )


            quantum_features.append(
                q
            )


        quantum_features = torch.stack(
            quantum_features
        ).float()


        logits = self.classifier(
            quantum_features
        )


        return logits


# ============================================================
# MODEL
# ============================================================

model = QuantumHead()

model = model.to(
    torch_device
)


criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    predictions = []
    labels_all = []


    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                torch_device
            )

            logits = model(
                x
            )

            pred = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(
                pred.cpu().numpy()
            )

            labels_all.extend(
                y.numpy()
            )


    return accuracy_score(
        labels_all,
        predictions
    )


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0

best_path = os.path.join(
    RESULT_DIR,
    "cifar10_quantum_head_best.pt"
)


train_accuracy_history = []
val_accuracy_history = []
loss_history = []


print("\n======================================")
print("TRAINING 8-QUBIT QUANTUM HEAD")
print("======================================")


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0
    total_correct = 0
    total = 0


    for batch_index, (
        x,
        y
    ) in enumerate(
        train_loader
    ):

        x = x.to(
            torch_device
        )

        y = y.to(
            torch_device
        )


        optimizer.zero_grad()


        logits = model(
            x
        )


        loss = criterion(
            logits,
            y
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            * y.size(0)
        )


        prediction = torch.argmax(
            logits,
            dim=1
        )


        total_correct += (
            prediction == y
        ).sum().item()


        total += y.size(0)


        if (
            batch_index + 1
        ) % 50 == 0:

            print(

                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "

                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"

            )


    train_loss = (
        total_loss
        / total
    )

    train_accuracy = (
        total_correct
        / total
    )


    val_accuracy = evaluate(
        model,
        val_loader
    )


    loss_history.append(
        train_loss
    )

    train_accuracy_history.append(
        train_accuracy
    )

    val_accuracy_history.append(
        val_accuracy
    )


    print(
        "\n--------------------------------------"
    )

    print(
        f"Epoch {epoch + 1}"
    )

    print(
        f"Loss           = "
        f"{train_loss:.4f}"
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
            best_path
        )

        print(
            "Best quantum model updated."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        best_path,
        map_location=torch_device
    )
)


# ============================================================
# FINAL TEST
# ============================================================

test_accuracy = evaluate(
    model,
    test_loader
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


# ============================================================
# SAVE FIGURE
# ============================================================

epochs = np.arange(
    1,
    len(
        val_accuracy_history
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
    "8-Qubit Hybrid Quantum Classifier on CIFAR-10"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


figure_path = os.path.join(
    FIGURE_DIR,
    "cifar10_quantum_head_accuracy.png"
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Figure saved to:",
    figure_path
)