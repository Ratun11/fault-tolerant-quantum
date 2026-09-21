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

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

N_QUBITS = 8
N_CLASSES = 10
N_UPLOADS = 4

# default.mixed is much heavier than default.qubit.
# Start with 64.
BATCH_SIZE = 64

NUM_WORKERS = 0


# ------------------------------------------------------------
# Noise sweep
#
# IMPORTANT:
# For the very first run, you may use only:
#
# NOISE_VALUES = [0.0]
#
# and verify that accuracy is close to 0.9143.
# ------------------------------------------------------------

NOISE_VALUES = [
    0.0,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050
]


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    "results/"
    "cifar10_resnet32_features.npz"
)

MODEL_PATH = (
    "results/"
    "cifar10_quantum_batched_best.pt"
)

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
# GPU DEVICE
#
# IMPORTANT:
#
# Quantum density-matrix simulation:
# CPU
#
# Classical classifier:
# GPU
# ============================================================

classifier_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n======================================")
print("DEVICE")
print("======================================")

print(
    "Classifier device:",
    classifier_device
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
        "PyTorch CUDA:",
        torch.version.cuda
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
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
print("TEST DATA")
print("======================================")

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# TORCH TEST DATASET
#
# Keep features on CPU because default.mixed runs on CPU.
# ============================================================

test_dataset = TensorDataset(

    torch.from_numpy(
        X_test
    ),

    torch.from_numpy(
        y_test
    )
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


# ============================================================
# QUANTUM DEVICE
#
# Noise channels require mixed-state simulation.
#
# For 8 qubits:
#
# statevector:
# 2^8 = 256
#
# density matrix:
# 256 x 256 = 65,536 entries
#
# Therefore this is substantially heavier than
# the ideal default.qubit simulation.
# ============================================================

qdev = qml.device(

    "default.mixed",

    wires=N_QUBITS,

    shots=None
)


# ============================================================
# NOISY QUANTUM CIRCUIT
#
# Same architecture that was trained previously:
#
# 32 ResNet features
#       ↓
# four groups of 8
#       ↓
# 8-qubit data re-uploading
#       ↓
# trainable RY/RZ
#       ↓
# CNOT ring
#
#
# Noise model:
#
# BitFlip(p) after:
#
# - data encoding RY
# - trainable RY
# - trainable RZ
# - each participating qubit after CNOT
#
#
# This is a controlled X-error model.
# It is NOT yet a realistic Rigetti hardware model.
# ============================================================

@qml.qnode(
    qdev,
    interface="torch",
    diff_method=None
)
def noisy_quantum_circuit(
    inputs,
    weights,
    noise_probability
):

    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )


        # ====================================================
        # 1. DATA ENCODING
        # ====================================================

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


            qml.BitFlip(

                noise_probability,

                wires=q
            )


        # ====================================================
        # 2. TRAINABLE ROTATIONS
        # ====================================================

        for q in range(
            N_QUBITS
        ):

            # RY
            qml.RY(

                weights[
                    upload,
                    q,
                    0
                ],

                wires=q
            )


            qml.BitFlip(

                noise_probability,

                wires=q
            )


            # RZ
            qml.RZ(

                weights[
                    upload,
                    q,
                    1
                ],

                wires=q
            )


            qml.BitFlip(

                noise_probability,

                wires=q
            )


        # ====================================================
        # 3. CNOT RING
        # ====================================================

        for q in range(
            N_QUBITS - 1
        ):

            qml.CNOT(

                wires=[
                    q,
                    q + 1
                ]
            )


            # Simplified CNOT-associated error
            # on both participating qubits

            qml.BitFlip(

                noise_probability,

                wires=q
            )


            qml.BitFlip(

                noise_probability,

                wires=q + 1
            )


        # ====================================================
        # CLOSE THE RING
        # ====================================================

        qml.CNOT(

            wires=[
                N_QUBITS - 1,
                0
            ]
        )


        qml.BitFlip(

            noise_probability,

            wires=N_QUBITS - 1
        )


        qml.BitFlip(

            noise_probability,

            wires=0
        )


    # ========================================================
    # QUANTUM OUTPUT
    #
    # 8 expectation values:
    #
    # <Z0>, ..., <Z7>
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
# QUANTUM CLASSIFIER
#
# Architecture must match the previously trained checkpoint.
# ============================================================

class QuantumClassifier(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        # ====================================================
        # QUANTUM PARAMETERS
        #
        # Keep these on CPU.
        # ====================================================

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


        # ====================================================
        # SAME CLASSICAL HEAD USED DURING TRAINING
        # ====================================================

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
        x,
        noise_probability
    ):

        # ====================================================
        # QUANTUM SIMULATION
        #
        # x:
        # CPU
        #
        # quantum_weights:
        # CPU
        #
        # output:
        # CPU
        # ====================================================

        q_outputs = (
            noisy_quantum_circuit(

                x,

                self.quantum_weights,

                noise_probability
            )
        )


        # ====================================================
        # CONVERT:
        #
        # tuple of 8 tensors
        #
        # ->
        #
        # [batch_size, 8]
        # ====================================================

        quantum_features = (
            torch.stack(

                q_outputs,

                dim=1
            )
            .float()
        )


        # ====================================================
        # MOVE ONLY THE 8-D QUANTUM OUTPUT TO GPU
        # ====================================================

        quantum_features = (
            quantum_features.to(
                classifier_device
            )
        )


        # ====================================================
        # CLASSICAL 8 -> 32 -> 10 HEAD ON GPU
        # ====================================================

        logits = self.classifier(
            quantum_features
        )


        return logits


# ============================================================
# CREATE MODEL
#
# Initially everything is CPU.
# ============================================================

model = QuantumClassifier()


# ============================================================
# LOAD TRAINED CHECKPOINT ON CPU
# ============================================================

checkpoint = torch.load(

    MODEL_PATH,

    map_location="cpu"
)


model.load_state_dict(
    checkpoint
)


# ============================================================
# DEVICE PLACEMENT
#
# Quantum parameters:
# CPU
#
# Classical classifier:
# CUDA
# ============================================================

model.quantum_weights.data = (
    model.quantum_weights
    .data
    .cpu()
)


model.classifier = (
    model.classifier.to(
        classifier_device
    )
)


# ============================================================
# FREEZE MODEL
#
# We are evaluating noise robustness only.
# No retraining.
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


model.eval()


print("\n======================================")
print("CHECKPOINT")
print("======================================")

print(
    "Loaded:",
    MODEL_PATH
)

print(
    "Quantum weights device:",
    model.quantum_weights.device
)

print(
    "Classifier device:",
    model.classifier[
        0
    ].weight.device
)

print(
    "Model frozen."
)


# ============================================================
# EVALUATE ONE NOISE LEVEL
# ============================================================

def evaluate_noise_level(
    p
):

    model.eval()


    all_predictions = []
    all_labels = []


    start_time = (
        time.time()
    )


    with torch.no_grad():

        for batch_index, (
            x,
            y
        ) in enumerate(
            test_loader
        ):


            # =================================================
            # IMPORTANT:
            #
            # Keep x on CPU.
            #
            # Do NOT send x to CUDA.
            # =================================================

            x = x.cpu()


            # =================================================
            # FORWARD
            # =================================================

            logits = model(

                x,

                float(p)
            )


            # =================================================
            # PREDICTION
            # =================================================

            predictions = (
                torch.argmax(

                    logits,

                    dim=1
                )
            )


            all_predictions.extend(

                predictions
                .detach()
                .cpu()
                .numpy()
            )


            all_labels.extend(
                y.numpy()
            )


            if (
                batch_index + 1
            ) % 20 == 0:

                print(

                    f"p={p:.4f} | "

                    f"Batch "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)}"
                )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(

        all_labels,

        all_predictions
    )


    macro_f1 = f1_score(

        all_labels,

        all_predictions,

        average="macro"
    )


    elapsed = (
        time.time()
        - start_time
    )


    return (
        accuracy,
        macro_f1,
        elapsed
    )


# ============================================================
# RUN NOISE SWEEP
# ============================================================

results = []


print("\n======================================")
print("QUANTUM BIT-FLIP NOISE SWEEP")
print("======================================")


for p in NOISE_VALUES:

    print(
        "\n--------------------------------------"
    )

    print(
        "Physical operation "
        "bit-flip probability =",
        p
    )


    (
        accuracy,
        macro_f1,
        elapsed
    ) = evaluate_noise_level(
        p
    )


    results.append({

        "p":
            p,

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "seconds":
            elapsed
    })


    print(
        f"\nAccuracy = "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1 = "
        f"{macro_f1:.4f}"
    )

    print(
        f"Time     = "
        f"{elapsed:.1f} seconds"
    )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n\n======================================")
print("FINAL RESULTS")
print("======================================")

print(
    "P_ERROR\t\tAccuracy\tMacro-F1"
)


for row in results:

    print(

        f"{row['p']:.4f}\t\t"

        f"{row['accuracy']:.4f}\t\t"

        f"{row['macro_f1']:.4f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

CSV_PATH = os.path.join(

    RESULT_DIR,

    "cifar10_quantum_noise_sweep.csv"
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

        "physical_bitflip_probability",

        "accuracy",

        "macro_f1",

        "evaluation_seconds"
    ])


    for row in results:

        writer.writerow([

            row["p"],

            row["accuracy"],

            row["macro_f1"],

            row["seconds"]
        ])


print(
    "\nResults saved to:",
    CSV_PATH
)


# ============================================================
# CONVERT RESULTS TO NUMPY
# ============================================================

p_values = np.array([

    row["p"]

    for row in results
])


accuracies = np.array([

    row["accuracy"]

    for row in results
])


macro_f1_values = np.array([

    row["macro_f1"]

    for row in results
])


# ============================================================
# IDEAL ACCURACY
#
# The p = 0 result from THIS SAME evaluation.
# ============================================================

ideal_accuracy = (
    accuracies[0]
)


ideal_f1 = (
    macro_f1_values[0]
)


# ============================================================
# FIGURE 1
# ACCURACY VS NOISE
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    accuracies * 100,

    marker="o",

    linewidth=2,

    label="Noisy quantum classifier"
)


plt.axhline(

    ideal_accuracy * 100,

    linestyle="--",

    label=(
        f"Ideal "
        f"({ideal_accuracy * 100:.2f}%)"
    )
)


plt.xlabel(
    "Physical Operation Bit-Flip Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Impact of Quantum Bit-Flip Noise on CIFAR-10"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_accuracy_vs_quantum_noise.png"
)


plt.savefig(

    ACCURACY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2
# MACRO-F1 VS NOISE
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    macro_f1_values * 100,

    marker="s",

    linewidth=2
)


plt.axhline(

    ideal_f1 * 100,

    linestyle="--"
)


plt.xlabel(
    "Physical Operation Bit-Flip Probability (%)"
)

plt.ylabel(
    "Macro-F1 (%)"
)

plt.title(
    "CIFAR-10 Macro-F1 Under Quantum Bit-Flip Noise"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


F1_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_f1_vs_quantum_noise.png"
)


plt.savefig(

    F1_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
# ACCURACY DROP
# ============================================================

accuracy_drop = (

    ideal_accuracy
    -
    accuracies
)


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    accuracy_drop * 100,

    marker="o",

    linewidth=2
)


plt.xlabel(
    "Physical Operation Bit-Flip Probability (%)"
)

plt.ylabel(
    "Accuracy Drop (percentage points)"
)

plt.title(
    "CIFAR-10 Accuracy Loss Due to Quantum Noise"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


DROP_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_accuracy_drop_vs_quantum_noise.png"
)


plt.savefig(

    DROP_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "CSV:",
    CSV_PATH
)

print(
    "Accuracy figure:",
    ACCURACY_FIGURE
)

print(
    "F1 figure:",
    F1_FIGURE
)

print(
    "Accuracy-drop figure:",
    DROP_FIGURE
)


print("\n======================================")
print("BASELINE CHECK")
print("======================================")

print(
    f"p = 0 accuracy = "
    f"{ideal_accuracy:.4f}"
)

print(
    "Previous ideal test accuracy "
    "was approximately 0.9143."
)