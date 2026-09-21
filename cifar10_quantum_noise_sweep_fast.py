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

# RTX 4090 has plenty of memory for 8-qubit statevectors
BATCH_SIZE = 512

# Monte-Carlo trajectories for noisy points.
#
# 8 is fine for development.
# Later we can use 32+ for final experiments.
N_TRAJECTORIES = 32

NUM_WORKERS = 0


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
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is intended for your RTX 4090."
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
# LOAD CIFAR-10 FEATURES
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


print("\n======================================")
print("DATA")
print("======================================")

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL DEFINITION
#
# This only exists so that we can load the previously trained
# checkpoint.
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
# LOAD TRAINED CHECKPOINT
# ============================================================

model = QuantumClassifier()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint
)

model = model.to(device)

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
    "Quantum weights:",
    model.quantum_weights.device
)

print(
    "Classifier:",
    model.classifier[0].weight.device
)


# ============================================================
# QUANTUM STATEVECTOR UTILITIES
#
# We directly simulate the same circuit in PyTorch.
#
# State:
#
# [batch, 2^8]
#
# = [batch, 256]
#
# complex64
# ============================================================


def initial_state(batch_size):

    state = torch.zeros(
        batch_size,
        2 ** N_QUBITS,
        dtype=torch.complex64,
        device=device
    )

    state[:, 0] = 1.0 + 0.0j

    return state


# ============================================================
# APPLY RY
#
# RY(theta) =
#
# [ cos(t/2)  -sin(t/2) ]
# [ sin(t/2)   cos(t/2) ]
#
# theta may be:
#
# scalar
# or
# [batch]
# ============================================================

def apply_ry(
    state,
    theta,
    wire
):

    batch_size = state.shape[0]

    left = 2 ** wire

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


    a = state_view[
        :,
        :,
        0,
        :
    ]

    b = state_view[
        :,
        :,
        1,
        :
    ]


    if not torch.is_tensor(theta):

        theta = torch.tensor(
            theta,
            device=device,
            dtype=torch.float32
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
        [
            out0,
            out1
        ],
        dim=2
    )


    return output.reshape(
        batch_size,
        -1
    )


# ============================================================
# APPLY RZ
#
# RZ(theta) =
#
# diag(
#     exp(-i theta/2),
#     exp(+i theta/2)
# )
# ============================================================

def apply_rz(
    state,
    theta,
    wire
):

    batch_size = state.shape[0]

    left = 2 ** wire

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


    a = state_view[
        :,
        :,
        0,
        :
    ]

    b = state_view[
        :,
        :,
        1,
        :
    ]


    if not torch.is_tensor(theta):

        theta = torch.tensor(
            theta,
            device=device,
            dtype=torch.float32
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
        -0.5j
        * theta
    )

    phase1 = torch.exp(
        0.5j
        * theta
    )


    out0 = phase0 * a
    out1 = phase1 * b


    output = torch.stack(
        [
            out0,
            out1
        ],
        dim=2
    )


    return output.reshape(
        batch_size,
        -1
    )


# ============================================================
# CONDITIONAL X ERROR
#
# mask:
#
# 0 -> identity
# 1 -> Pauli X
#
# shape:
#
# [batch]
# ============================================================

def apply_x_fault(
    state,
    mask,
    wire
):

    batch_size = state.shape[0]

    left = 2 ** wire

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


    a = state_view[
        :,
        :,
        0,
        :
    ]

    b = state_view[
        :,
        :,
        1,
        :
    ]


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
        [
            out0,
            out1
        ],
        dim=2
    )


    return output.reshape(
        batch_size,
        -1
    )


# ============================================================
# BUILD CNOT PERMUTATIONS
#
# CNOT is implemented by permuting the statevector.
#
# This is precomputed once.
# ============================================================

def build_cnot_permutation(
    control,
    target
):

    indices = torch.arange(
        2 ** N_QUBITS,
        device=device,
        dtype=torch.long
    )


    # wire 0 = most significant bit

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


# ============================================================
# APPLY CNOT
# ============================================================

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
# Z EXPECTATION VALUES
#
# Precompute eigenvalues:
#
# +1 for |0>
# -1 for |1>
# ============================================================

basis_indices = torch.arange(
    2 ** N_QUBITS,
    device=device,
    dtype=torch.long
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
        2.0
        * bit.float()
    )


    z_signs.append(
        signs
    )


# [256, 8]

Z_SIGN_MATRIX = torch.stack(
    z_signs,
    dim=1
)


# ============================================================
# RANDOM BIT-FLIP MASK
# ============================================================

def random_fault_mask(
    shape,
    p
):

    if p <= 0:

        return torch.zeros(
            shape,
            device=device,
            dtype=torch.bool
        )


    return (
        torch.rand(
            shape,
            device=device
        )
        <
        p
    )


# ============================================================
# FAST 8-QUBIT CIRCUIT
#
# This reproduces:
#
# for each upload:
#
#   RY(data)
#   X noise
#
#   RY(weight)
#   X noise
#
#   RZ(weight)
#   X noise
#
#   CNOT ring
#   X noise on both CNOT qubits
#
#
# This is the same controlled bit-flip model used in the
# earlier PennyLane noisy script.
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    # ========================================================
    # RANDOM FAULT MASKS
    # ========================================================

    encoding_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p
    )


    ry_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p
    )


    rz_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p
    )


    # Last dimension:
    #
    # 0 = control error
    # 1 = target error

    cnot_faults = random_fault_mask(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2
        ),

        p
    )


    # ========================================================
    # CIRCUIT
    # ========================================================

    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )


        # ====================================================
        # DATA ENCODING
        # ====================================================

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


        # ====================================================
        # TRAINABLE RY + RZ
        # ====================================================

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


        # ====================================================
        # CNOT RING
        # ====================================================

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


            # Control fault

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


            # Target fault

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


    # ========================================================
    # EXPECTATION VALUES
    #
    # probabilities:
    #
    # [batch, 256]
    #
    # output:
    #
    # [batch, 8]
    # ========================================================

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
# EVALUATE ONE NOISE LEVEL
# ============================================================

def evaluate_noise(
    p,
    noise_index
):

    predictions_all = []
    labels_all = []


    # --------------------------------------------------------
    # Deterministic but different random noise pattern for
    # each p.
    # --------------------------------------------------------

    torch.manual_seed(
        SEED + noise_index
    )

    torch.cuda.manual_seed_all(
        SEED + noise_index
    )


    torch.cuda.synchronize()

    start_time = time.time()


    with torch.no_grad():

        for batch_index, (
            x,
            y
        ) in enumerate(
            test_loader
        ):


            x = x.to(
                device,
                non_blocking=True
            )


            original_batch_size = (
                x.shape[0]
            )


            # =================================================
            # p = 0 needs only one trajectory.
            # =================================================

            if p == 0:

                trajectories = 1

            else:

                trajectories = (
                    N_TRAJECTORIES
                )


            # =================================================
            # REPEAT EACH IMAGE T TIMES
            #
            # [B, 32]
            #
            # ->
            #
            # [B, T, 32]
            #
            # ->
            #
            # [B*T, 32]
            # =================================================

            expanded_x = (

                x
                .unsqueeze(1)

                .expand(
                    -1,
                    trajectories,
                    -1
                )

                .reshape(
                    original_batch_size
                    * trajectories,
                    32
                )
            )


            # =================================================
            # QUANTUM CIRCUIT
            #
            # Entire operation stays on RTX 4090.
            # =================================================

            quantum_features = (
                run_quantum_circuit(

                    expanded_x,

                    model.quantum_weights,

                    p
                )
            )


            # =================================================
            # MONTE-CARLO AVERAGE
            #
            # Average <Z> across trajectories.
            #
            # This approximates the expectation values of
            # the corresponding bit-flip quantum channel.
            # =================================================

            if trajectories > 1:

                quantum_features = (

                    quantum_features

                    .reshape(
                        original_batch_size,
                        trajectories,
                        N_QUBITS
                    )

                    .mean(
                        dim=1
                    )
                )


            # =================================================
            # CLASSICAL HEAD
            # =================================================

            logits = model.classifier(
                quantum_features
            )


            prediction = torch.argmax(
                logits,
                dim=1
            )


            predictions_all.extend(

                prediction
                .cpu()
                .numpy()
            )


            labels_all.extend(
                y.numpy()
            )


            if (
                batch_index + 1
            ) % 5 == 0:

                print(

                    f"p={p:.4f} | "

                    f"Batch "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)}"
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
# RUN NOISE SWEEP
# ============================================================

results = []


print("\n======================================")
print("GPU MONTE-CARLO QUANTUM NOISE SWEEP")
print("======================================")

print(
    "Trajectories per noisy sample:",
    N_TRAJECTORIES
)


for noise_index, p in enumerate(
    NOISE_VALUES
):

    print(
        "\n--------------------------------------"
    )

    print(
        f"Physical bit-flip probability = "
        f"{p}"
    )


    (
        accuracy,
        macro_f1,
        elapsed
    ) = evaluate_noise(
        p,
        noise_index
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
        f"{elapsed:.2f} sec"
    )


    # ========================================================
    # IMPORTANT BASELINE CHECK
    # ========================================================

    if p == 0:

        print(
            "\nBaseline check:"
        )

        print(
            "Expected approximately: "
            "0.9143"
        )

        print(
            "Obtained:",
            f"{accuracy:.4f}"
        )


        if abs(
            accuracy - 0.9143
        ) > 0.01:

            print(
                "\nWARNING:"
            )

            print(
                "p=0 differs noticeably "
                "from the original quantum model."
            )

            print(
                "Do not interpret noisy results "
                "until the simulator is checked."
            )


# ============================================================
# FINAL TABLE
# ============================================================

print("\n======================================")
print("FINAL RESULTS")
print("======================================")

print(
    "P_ERROR\t\tAccuracy\tMacro-F1\tSeconds"
)


for row in results:

    print(

        f"{row['p']:.4f}\t\t"

        f"{row['accuracy']:.4f}\t\t"

        f"{row['macro_f1']:.4f}\t\t"

        f"{row['seconds']:.2f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

CSV_PATH = os.path.join(

    RESULT_DIR,

    "cifar10_quantum_noise_sweep_gpu.csv"
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

        "trajectories",

        "evaluation_seconds"
    ])


    for row in results:

        writer.writerow([

            row["p"],

            row["accuracy"],

            row["macro_f1"],

            (
                1
                if row["p"] == 0
                else N_TRAJECTORIES
            ),

            row["seconds"]
        ])


# ============================================================
# NUMPY ARRAYS FOR PLOTTING
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


ideal_accuracy = (
    accuracies[0]
)


# ============================================================
# FIGURE 1:
# TEST ACCURACY VS NOISE
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
    "Physical Bit-Flip Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "CIFAR-10 Accuracy Under Quantum Bit-Flip Noise"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_accuracy_vs_quantum_noise_gpu.png"
)


plt.savefig(

    ACCURACY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2:
# MACRO F1 VS NOISE
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    macro_f1_values * 100,

    marker="o",

    linewidth=2
)


plt.xlabel(
    "Physical Bit-Flip Probability (%)"
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

    "cifar10_f1_vs_quantum_noise_gpu.png"
)


plt.savefig(

    F1_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3:
# ACCURACY DROP FROM IDEAL
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
    "Physical Bit-Flip Probability (%)"
)

plt.ylabel(
    "Accuracy Drop (percentage points)"
)

plt.title(
    "Accuracy Degradation from Quantum Bit-Flip Noise"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


DROP_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_accuracy_drop_quantum_noise_gpu.png"
)


plt.savefig(

    DROP_FIGURE,

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