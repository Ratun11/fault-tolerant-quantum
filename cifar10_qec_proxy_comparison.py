import os
import csv
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

N_QUBITS = 8
N_CLASSES = 10
N_UPLOADS = 4

BATCH_SIZE = 512

# You said you are now using 32.
N_TRAJECTORIES = 32

# Repeat Monte-Carlo experiment with different random seeds.
N_SEEDS = 5

NOISE_VALUES = [
    0.0,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050
]


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
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required for this script."
    )


device = torch.device("cuda")


print("\n======================================")
print("DEVICE")
print("======================================")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)


# ============================================================
# DATA
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
)


X_test = data[
    "X_test"
].astype(np.float32)

y_test = data[
    "y_test"
].astype(np.int64)


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

    num_workers=0,

    pin_memory=True
)


print("\n======================================")
print("TEST DATA")
print("======================================")

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# TRAINED MODEL DEFINITION
# ============================================================

class QuantumClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.quantum_weights = nn.Parameter(

            0.05 * torch.randn(
                N_UPLOADS,
                N_QUBITS,
                2
            )
        )

        self.classifier = nn.Sequential(

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


# ============================================================
# LOAD CHECKPOINT
# ============================================================

model = QuantumClassifier()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint
)

model = model.to(
    device
)

model.eval()


for parameter in model.parameters():
    parameter.requires_grad = False


print("\n======================================")
print("CHECKPOINT")
print("======================================")

print(
    "Loaded:",
    MODEL_PATH
)

print(
    "Quantum weights:",
    model.quantum_weights.device
)

print(
    "Classifier:",
    model.classifier[0].weight.device
)


# ============================================================
# STATE INITIALIZATION
#
# 8 qubits:
#
# state shape =
#
# [batch, 256]
# ============================================================

def initial_state(
    batch_size
):

    state = torch.zeros(

        batch_size,

        2 ** N_QUBITS,

        dtype=torch.complex64,

        device=device
    )

    state[:, 0] = (
        1.0 + 0.0j
    )

    return state


# ============================================================
# RY
# ============================================================

def apply_ry(
    state,
    theta,
    wire
):

    batch_size = (
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2 ** (
            N_QUBITS
            - wire
            - 1
        )
    )


    state_view = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = state_view[:, :, 0, :]
    b = state_view[:, :, 1, :]


    if not torch.is_tensor(theta):

        theta = torch.tensor(
            theta,
            dtype=torch.float32,
            device=device
        )


    theta = theta.to(
        device=device,
        dtype=torch.float32
    )


    if theta.ndim == 0:

        theta = theta.expand(
            batch_size
        )


    theta = theta.reshape(
        batch_size,
        1,
        1
    )


    c = torch.cos(
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )


    out0 = (
        c * a
        -
        s * b
    )

    out1 = (
        s * a
        +
        c * b
    )


    output = torch.stack(
        [out0, out1],
        dim=2
    )


    return output.reshape(
        batch_size,
        -1
    )


# ============================================================
# RZ
# ============================================================

def apply_rz(
    state,
    theta,
    wire
):

    batch_size = (
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2 ** (
            N_QUBITS
            - wire
            - 1
        )
    )


    state_view = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = state_view[:, :, 0, :]
    b = state_view[:, :, 1, :]


    if not torch.is_tensor(theta):

        theta = torch.tensor(
            theta,
            dtype=torch.float32,
            device=device
        )


    theta = theta.to(
        device=device,
        dtype=torch.float32
    )


    if theta.ndim == 0:

        theta = theta.expand(
            batch_size
        )


    theta = theta.reshape(
        batch_size,
        1,
        1
    )


    phase0 = torch.exp(
        -0.5j * theta
    )

    phase1 = torch.exp(
        0.5j * theta
    )


    out0 = phase0 * a
    out1 = phase1 * b


    output = torch.stack(
        [out0, out1],
        dim=2
    )


    return output.reshape(
        batch_size,
        -1
    )


# ============================================================
# CONDITIONAL X FAULT
# ============================================================

def apply_x_fault(
    state,
    mask,
    wire
):

    batch_size = (
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2 ** (
            N_QUBITS
            - wire
            - 1
        )
    )


    state_view = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = state_view[:, :, 0, :]
    b = state_view[:, :, 1, :]


    mask = mask.bool().reshape(
        batch_size,
        1,
        1
    )


    out0 = torch.where(
        mask,
        b,
        a
    )

    out1 = torch.where(
        mask,
        a,
        b
    )


    output = torch.stack(
        [out0, out1],
        dim=2
    )


    return output.reshape(
        batch_size,
        -1
    )


# ============================================================
# CNOT PERMUTATIONS
# ============================================================

def build_cnot_permutation(
    control,
    target
):

    indices = torch.arange(

        2 ** N_QUBITS,

        dtype=torch.long,

        device=device
    )


    control_shift = (
        N_QUBITS
        - 1
        - control
    )

    target_shift = (
        N_QUBITS
        - 1
        - target
    )


    control_bit = (

        indices
        >> control_shift

    ) & 1


    permutation = (

        indices

        ^

        (
            control_bit
            << target_shift
        )
    )


    return permutation


CNOT_PERMUTATIONS = {}


for q in range(
    N_QUBITS
):

    control = q

    target = (
        q + 1
    ) % N_QUBITS


    CNOT_PERMUTATIONS[
        (control, target)
    ] = build_cnot_permutation(
        control,
        target
    )


def apply_cnot(
    state,
    control,
    target
):

    permutation = (
        CNOT_PERMUTATIONS[
            (control, target)
        ]
    )

    return state[
        :,
        permutation
    ]


# ============================================================
# PRECOMPUTE Z EXPECTATION SIGNS
# ============================================================

basis_indices = torch.arange(

    2 ** N_QUBITS,

    dtype=torch.long,

    device=device
)


z_signs = []


for q in range(
    N_QUBITS
):

    shift = (
        N_QUBITS
        - 1
        - q
    )


    bit = (

        basis_indices
        >> shift

    ) & 1


    signs = (

        1.0
        -
        2.0 * bit.float()
    )


    z_signs.append(
        signs
    )


Z_SIGN_MATRIX = torch.stack(
    z_signs,
    dim=1
)


# ============================================================
# RANDOM X FAULTS
# ============================================================

def random_fault_mask(
    shape,
    error_probability
):

    if error_probability <= 0:

        return torch.zeros(

            shape,

            dtype=torch.bool,

            device=device
        )


    return (

        torch.rand(
            shape,
            device=device
        )

        < error_probability
    )


# ============================================================
# QUANTUM CIRCUIT
#
# Same tested GPU simulator from the previous experiment.
#
# Total modeled physical X-error opportunities:
#
# each upload:
#
# 8 encoding RY
# 8 trainable RY
# 8 trainable RZ
# 8 CNOT x 2 qubits
#
# = 40
#
# 4 uploads:
#
# = 160 fault locations
# ============================================================

N_FAULT_LOCATIONS = (
    N_UPLOADS
    *
    (
        N_QUBITS
        +
        N_QUBITS
        +
        N_QUBITS
        +
        2 * N_QUBITS
    )
)


def run_quantum_circuit(
    inputs,
    quantum_weights,
    error_probability
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    encoding_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        error_probability
    )


    ry_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        error_probability
    )


    rz_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        error_probability
    )


    cnot_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2
        ),

        error_probability
    )


    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )


        # ----------------------------------------------------
        # Data encoding
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            state = apply_ry(

                state,

                inputs[
                    :,
                    start + q
                ],

                q
            )


            state = apply_x_fault(

                state,

                encoding_faults[
                    :,
                    upload,
                    q
                ],

                q
            )


        # ----------------------------------------------------
        # Trainable rotations
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            state = apply_ry(

                state,

                quantum_weights[
                    upload,
                    q,
                    0
                ],

                q
            )


            state = apply_x_fault(

                state,

                ry_faults[
                    :,
                    upload,
                    q
                ],

                q
            )


            state = apply_rz(

                state,

                quantum_weights[
                    upload,
                    q,
                    1
                ],

                q
            )


            state = apply_x_fault(

                state,

                rz_faults[
                    :,
                    upload,
                    q
                ],

                q
            )


        # ----------------------------------------------------
        # CNOT ring
        # ----------------------------------------------------

        for gate_index in range(
            N_QUBITS
        ):

            control = (
                gate_index
            )

            target = (
                gate_index + 1
            ) % N_QUBITS


            state = apply_cnot(

                state,

                control,

                target
            )


            state = apply_x_fault(

                state,

                cnot_faults[
                    :,
                    upload,
                    gate_index,
                    0
                ],

                control
            )


            state = apply_x_fault(

                state,

                cnot_faults[
                    :,
                    upload,
                    gate_index,
                    1
                ],

                target
            )


    probabilities = (

        state.real.square()

        +

        state.imag.square()
    )


    expectations = (

        probabilities

        @

        Z_SIGN_MATRIX
    )


    return expectations


# ============================================================
# 3-QUBIT REPETITION CODE
#
# Physical error:
#
# p
#
# Logical failure occurs when 2 or 3 physical qubits flip:
#
# pL =
#
# 3 p^2 (1-p) + p^3
#
# =
#
# 3p^2 - 2p^3
#
# This assumes PERFECT syndrome extraction and recovery.
# ============================================================

def repetition_logical_error(
    p
):

    return (
        3.0 * p ** 2
        -
        2.0 * p ** 3
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    error_probability,
    random_seed
):

    torch.manual_seed(
        random_seed
    )

    torch.cuda.manual_seed_all(
        random_seed
    )


    predictions_all = []
    labels_all = []


    torch.cuda.synchronize()

    start_time = (
        time.time()
    )


    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(
                device,
                non_blocking=True
            )


            batch_size = (
                x.shape[0]
            )


            if error_probability == 0:

                trajectories = 1

            else:

                trajectories = (
                    N_TRAJECTORIES
                )


            # ------------------------------------------------
            # Repeat each sample for Monte-Carlo trajectories
            # ------------------------------------------------

            expanded_x = (

                x.unsqueeze(1)

                .expand(
                    -1,
                    trajectories,
                    -1
                )

                .reshape(
                    batch_size
                    * trajectories,
                    32
                )
            )


            # ------------------------------------------------
            # Quantum simulation
            # ------------------------------------------------

            quantum_features = (
                run_quantum_circuit(

                    expanded_x,

                    model.quantum_weights,

                    error_probability
                )
            )


            # ------------------------------------------------
            # Average trajectories
            # ------------------------------------------------

            if trajectories > 1:

                quantum_features = (

                    quantum_features

                    .reshape(
                        batch_size,
                        trajectories,
                        N_QUBITS
                    )

                    .mean(
                        dim=1
                    )
                )


            # ------------------------------------------------
            # Classical output head
            # ------------------------------------------------

            logits = model.classifier(
                quantum_features
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            predictions_all.extend(

                predictions
                .cpu()
                .numpy()
            )


            labels_all.extend(
                y.numpy()
            )


    torch.cuda.synchronize()


    elapsed = (
        time.time()
        -
        start_time
    )


    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )


    macro_f1 = f1_score(

        labels_all,

        predictions_all,

        average="macro"
    )


    return (
        accuracy,
        macro_f1,
        elapsed
    )


# ============================================================
# IDEAL BASELINE
# ============================================================

print("\n======================================")
print("IDEAL BASELINE")
print("======================================")


ideal_accuracy, ideal_f1, ideal_time = evaluate(

    error_probability=0.0,

    random_seed=SEED
)


print(
    f"Ideal accuracy = "
    f"{ideal_accuracy:.4f}"
)

print(
    f"Ideal Macro-F1 = "
    f"{ideal_f1:.4f}"
)

print(
    f"Expected       ≈ "
    f"0.9143"
)


# ============================================================
# EXPERIMENT
# ============================================================

detailed_results = []


print("\n======================================")
print("QEC PROXY EXPERIMENT")
print("======================================")

print(
    "Monte-Carlo trajectories:",
    N_TRAJECTORIES
)

print(
    "Independent seeds:",
    N_SEEDS
)

print(
    "Modeled fault locations:",
    N_FAULT_LOCATIONS
)


for p_index, physical_p in enumerate(
    NOISE_VALUES
):

    logical_p = (
        repetition_logical_error(
            physical_p
        )
    )


    print(
        "\n======================================"
    )

    print(
        f"Physical p = "
        f"{physical_p:.6f}"
    )

    print(
        f"QEC logical p = "
        f"{logical_p:.8f}"
    )

    print(
        "======================================"
    )


    for seed_index in range(
        N_SEEDS
    ):

        current_seed = (
            SEED
            +
            1000 * p_index
            +
            seed_index
        )


        # ====================================================
        # UNPROTECTED
        # ====================================================

        if physical_p == 0:

            unprotected_accuracy = (
                ideal_accuracy
            )

            unprotected_f1 = (
                ideal_f1
            )

            unprotected_time = 0.0

        else:

            (
                unprotected_accuracy,
                unprotected_f1,
                unprotected_time

            ) = evaluate(

                physical_p,

                current_seed
            )


        # ====================================================
        # IDEAL QEC PROXY
        # ====================================================

        if physical_p == 0:

            protected_accuracy = (
                ideal_accuracy
            )

            protected_f1 = (
                ideal_f1
            )

            protected_time = 0.0

        else:

            (
                protected_accuracy,
                protected_f1,
                protected_time

            ) = evaluate(

                logical_p,

                current_seed + 100000
            )


        detailed_results.append({

            "physical_p":
                physical_p,

            "logical_p":
                logical_p,

            "seed":
                current_seed,

            "unprotected_accuracy":
                unprotected_accuracy,

            "protected_accuracy":
                protected_accuracy,

            "unprotected_f1":
                unprotected_f1,

            "protected_f1":
                protected_f1,

            "unprotected_time":
                unprotected_time,

            "protected_time":
                protected_time
        })


        print(

            f"Seed {seed_index + 1}/{N_SEEDS} | "

            f"Unprotected="
            f"{unprotected_accuracy:.4f} | "

            f"QEC="
            f"{protected_accuracy:.4f}"
        )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

DETAILED_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_qec_proxy_detailed.csv"
)


with open(
    DETAILED_CSV,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([

        "physical_p",
        "logical_p",
        "seed",

        "unprotected_accuracy",
        "protected_accuracy",

        "unprotected_macro_f1",
        "protected_macro_f1",

        "unprotected_seconds",
        "protected_seconds"
    ])


    for row in detailed_results:

        writer.writerow([

            row["physical_p"],
            row["logical_p"],
            row["seed"],

            row["unprotected_accuracy"],
            row["protected_accuracy"],

            row["unprotected_f1"],
            row["protected_f1"],

            row["unprotected_time"],
            row["protected_time"]
        ])


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = []


for physical_p in NOISE_VALUES:

    subset = [

        row

        for row in detailed_results

        if row["physical_p"]
        == physical_p
    ]


    logical_p = (
        subset[0]["logical_p"]
    )


    unprotected_acc = np.array([

        row["unprotected_accuracy"]

        for row in subset
    ])


    protected_acc = np.array([

        row["protected_accuracy"]

        for row in subset
    ])


    unprotected_f1 = np.array([

        row["unprotected_f1"]

        for row in subset
    ])


    protected_f1 = np.array([

        row["protected_f1"]

        for row in subset
    ])


    summary.append({

        "physical_p":
            physical_p,

        "logical_p":
            logical_p,

        "unprotected_acc_mean":
            unprotected_acc.mean(),

        "unprotected_acc_std":
            unprotected_acc.std(ddof=1)
            if len(unprotected_acc) > 1
            else 0.0,

        "protected_acc_mean":
            protected_acc.mean(),

        "protected_acc_std":
            protected_acc.std(ddof=1)
            if len(protected_acc) > 1
            else 0.0,

        "unprotected_f1_mean":
            unprotected_f1.mean(),

        "protected_f1_mean":
            protected_f1.mean()
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

SUMMARY_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_qec_proxy_summary.csv"
)


with open(
    SUMMARY_CSV,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([

        "physical_p",

        "logical_p",

        "expected_physical_faults",

        "expected_logical_faults",

        "unprotected_accuracy_mean",

        "unprotected_accuracy_std",

        "protected_accuracy_mean",

        "protected_accuracy_std",

        "unprotected_macro_f1_mean",

        "protected_macro_f1_mean"
    ])


    for row in summary:

        writer.writerow([

            row["physical_p"],

            row["logical_p"],

            N_FAULT_LOCATIONS
            * row["physical_p"],

            N_FAULT_LOCATIONS
            * row["logical_p"],

            row[
                "unprotected_acc_mean"
            ],

            row[
                "unprotected_acc_std"
            ],

            row[
                "protected_acc_mean"
            ],

            row[
                "protected_acc_std"
            ],

            row[
                "unprotected_f1_mean"
            ],

            row[
                "protected_f1_mean"
            ]
        ])


# ============================================================
# PRINT SUMMARY TABLE
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p\t\tp_L\t\t"
    "Unprotected\tQEC Proxy"
)


for row in summary:

    print(

        f"{row['physical_p']:.4f}\t\t"

        f"{row['logical_p']:.6f}\t"

        f"{row['unprotected_acc_mean']:.4f}"
        f" ± "
        f"{row['unprotected_acc_std']:.4f}\t"

        f"{row['protected_acc_mean']:.4f}"
        f" ± "
        f"{row['protected_acc_std']:.4f}"
    )


# ============================================================
# PLOT DATA
# ============================================================

p_values = np.array([

    row["physical_p"]

    for row in summary
])


unprotected_mean = np.array([

    row["unprotected_acc_mean"]

    for row in summary
])


unprotected_std = np.array([

    row["unprotected_acc_std"]

    for row in summary
])


protected_mean = np.array([

    row["protected_acc_mean"]

    for row in summary
])


protected_std = np.array([

    row["protected_acc_std"]

    for row in summary
])


# ============================================================
# FIGURE 1
# ACCURACY COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5.8)
)


plt.axhline(

    ideal_accuracy * 100,

    linestyle="--",

    label=(
        f"Ideal "
        f"({ideal_accuracy * 100:.2f}%)"
    )
)


plt.plot(

    p_values * 100,

    unprotected_mean * 100,

    marker="o",

    linewidth=2,

    label="Unprotected noisy QML"
)


plt.fill_between(

    p_values * 100,

    (
        unprotected_mean
        -
        unprotected_std
    ) * 100,

    (
        unprotected_mean
        +
        unprotected_std
    ) * 100,

    alpha=0.2
)


plt.plot(

    p_values * 100,

    protected_mean * 100,

    marker="s",

    linewidth=2,

    label="3-qubit ideal-QEC proxy"
)


plt.fill_between(

    p_values * 100,

    (
        protected_mean
        -
        protected_std
    ) * 100,

    (
        protected_mean
        +
        protected_std
    ) * 100,

    alpha=0.2
)


plt.xlabel(
    "Physical Bit-Flip Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Error-Corrected Quantum Classification on CIFAR-10"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_qec_proxy_accuracy_comparison.png"
)


plt.savefig(

    ACCURACY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2
# PHYSICAL VS LOGICAL ERROR RATE
# ============================================================

logical_values = np.array([

    row["logical_p"]

    for row in summary
])


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    p_values * 100,

    marker="o",

    linewidth=2,

    label="Unprotected"
)


plt.plot(

    p_values * 100,

    logical_values * 100,

    marker="s",

    linewidth=2,

    label="3-qubit repetition code"
)


plt.xlabel(
    "Physical Bit-Flip Probability (%)"
)

plt.ylabel(
    "Effective Error Probability (%)"
)

plt.title(
    "Physical-to-Logical Error Suppression"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ERROR_FIGURE = os.path.join(

    FIGURE_DIR,

    "repetition_code_physical_vs_logical_error.png"
)


plt.savefig(

    ERROR_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
# ACCURACY RECOVERY
# ============================================================

recovery = (

    protected_mean
    -
    unprotected_mean
)


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    recovery * 100,

    marker="o",

    linewidth=2
)


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Physical Bit-Flip Probability (%)"
)

plt.ylabel(
    "Accuracy Recovered (percentage points)"
)

plt.title(
    "Classification Accuracy Recovered by Ideal QEC"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


RECOVERY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_qec_accuracy_recovery.png"
)


plt.savefig(

    RECOVERY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Detailed CSV:",
    DETAILED_CSV
)

print(
    "Summary CSV:",
    SUMMARY_CSV
)

print(
    "Accuracy comparison:",
    ACCURACY_FIGURE
)

print(
    "Physical/logical error:",
    ERROR_FIGURE
)

print(
    "Accuracy recovery:",
    RECOVERY_FIGURE
)


print("\n======================================")
print("IMPORTANT INTERPRETATION")
print("======================================")

print(
    "This experiment uses an IDEALIZED "
    "3-qubit repetition-code QEC proxy."
)

print(
    "It assumes perfect encoding, "
    "syndrome extraction, decoding, "
    "and recovery."
)

print(
    "It protects only against "
    "X / bit-flip errors."
)

print(
    "It is not yet a full encoded "
    "fault-tolerant implementation."
)