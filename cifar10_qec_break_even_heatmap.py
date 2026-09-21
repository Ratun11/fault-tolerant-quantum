import os
import csv
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

N_QUBITS = 8
N_CLASSES = 10
N_UPLOADS = 4

BATCH_SIZE = 512

# Keep your current value
N_TRAJECTORIES = 32

# Average the classifier calibration over two seeds
N_CALIBRATION_SEEDS = 2

# Samples used to estimate logical error probability
BLOCK_SAMPLES = 500_000


# ============================================================
# PHYSICAL DATA ERROR VALUES
#
# One heatmap will be produced for each.
# ============================================================

P_DATA_VALUES = [
    0.005,
    0.010,
    0.020,
    0.050
]


# ============================================================
# SYNDROME-CIRCUIT GRID
# ============================================================

P_CNOT_VALUES = [
    0.000,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020
]


P_MEAS_VALUES = [
    0.000,
    0.005,
    0.010,
    0.020,
    0.050
]


# Keep recovery noise fixed for this experiment
P_RECOVERY = 0.005


# ============================================================
# CLASSIFIER CALIBRATION GRID
#
# We first measure:
#
# effective logical X-error probability
#              ->
# CIFAR-10 accuracy
#
# Then use interpolation for the QEC heatmaps.
# ============================================================

CALIBRATION_P = np.unique(
    np.concatenate([
        np.linspace(
            0.000,
            0.010,
            11
        ),

        np.linspace(
            0.0125,
            0.080,
            28
        )
    ])
)


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
# GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required."
    )


device = torch.device("cuda")


print("\n======================================")
print("DEVICE")
print("======================================")

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA:",
    torch.version.cuda
)


# ============================================================
# LOAD CIFAR-10 FEATURES
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
print("DATA")
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
# TRAINED MODEL
# ============================================================

class QuantumClassifier(nn.Module):

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


        self.classifier = nn.Sequential(

            nn.Linear(
                N_QUBITS,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                32,
                N_CLASSES
            )
        )


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
# STATEVECTOR
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


    x = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


    if not torch.is_tensor(theta):

        theta = torch.tensor(
            theta,
            dtype=torch.float32,
            device=device
        )


    theta = theta.to(
        dtype=torch.float32,
        device=device
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


    return torch.stack(

        [
            out0,
            out1
        ],

        dim=2

    ).reshape(
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


    x = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


    if not torch.is_tensor(theta):

        theta = torch.tensor(
            theta,
            dtype=torch.float32,
            device=device
        )


    theta = theta.to(
        dtype=torch.float32,
        device=device
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


    return torch.stack(

        [
            phase0 * a,
            phase1 * b
        ],

        dim=2

    ).reshape(
        batch_size,
        -1
    )


# ============================================================
# X FAULT
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


    x = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


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


    return torch.stack(

        [
            out0,
            out1
        ],

        dim=2

    ).reshape(
        batch_size,
        -1
    )


# ============================================================
# CNOT
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


    return (

        indices

        ^

        (
            control_bit
            << target_shift
        )
    )


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

    return state[
        :,
        CNOT_PERMUTATIONS[
            (control, target)
        ]
    ]


# ============================================================
# Z EXPECTATION MATRIX
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
# RANDOM FAULT
# ============================================================

def bernoulli_fault(
    shape,
    p
):

    if p <= 0:

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

        < p
    )


# ============================================================
# UNPROTECTED LOGICAL-NOISE CIRCUIT
#
# Used to calibrate:
#
# logical X-error p
#        ->
# classification accuracy
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p_error
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    encoding_faults = bernoulli_fault(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p_error
    )


    ry_faults = bernoulli_fault(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p_error
    )


    rz_faults = bernoulli_fault(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p_error
    )


    cnot_faults = bernoulli_fault(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2
        ),

        p_error
    )


    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )


        # ----------------------------------------------------
        # Encoding
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
        # Trainable gates
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

            control = gate_index

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


    return (

        probabilities

        @

        Z_SIGN_MATRIX
    )


# ============================================================
# CLASSIFIER ACCURACY FOR A GIVEN LOGICAL ERROR RATE
# ============================================================

def evaluate_logical_error(
    p_error,
    random_seed
):

    torch.manual_seed(
        random_seed
    )

    torch.cuda.manual_seed_all(
        random_seed
    )


    predictions = []
    labels = []


    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(
                device,
                non_blocking=True
            )


            batch_size = (
                x.shape[0]
            )


            trajectories = (

                1

                if p_error == 0

                else N_TRAJECTORIES
            )


            expanded_x = (

                x
                .unsqueeze(1)

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


            q_features = (
                run_quantum_circuit(

                    expanded_x,

                    model.quantum_weights,

                    p_error
                )
            )


            if trajectories > 1:

                q_features = (

                    q_features

                    .reshape(
                        batch_size,
                        trajectories,
                        N_QUBITS
                    )

                    .mean(
                        dim=1
                    )
                )


            logits = model.classifier(
                q_features
            )


            pred = torch.argmax(
                logits,
                dim=1
            )


            predictions.extend(

                pred
                .cpu()
                .numpy()
            )


            labels.extend(
                y.numpy()
            )


    return accuracy_score(
        labels,
        predictions
    )


# ============================================================
# CIRCUIT-LEVEL QEC BLOCK SIMULATOR
# ============================================================

def noisy_syndrome_cnot(
    data_bit,
    ancilla_bit,
    p_cnot
):

    # Ideal parity propagation
    ancilla_bit = torch.logical_xor(
        ancilla_bit,
        data_bit
    )


    # Fault on data participant
    data_bit = torch.logical_xor(

        data_bit,

        bernoulli_fault(
            data_bit.shape,
            p_cnot
        )
    )


    # Fault on ancilla participant
    ancilla_bit = torch.logical_xor(

        ancilla_bit,

        bernoulli_fault(
            ancilla_bit.shape,
            p_cnot
        )
    )


    return (
        data_bit,
        ancilla_bit
    )


# ============================================================
# SAMPLE LOGICAL FAILURE OF REPETITION-CODE BLOCK
# ============================================================

def sample_qec_logical_failure(
    samples,
    p_data,
    p_cnot,
    p_meas,
    p_recovery,
    rounds
):

    # --------------------------------------------------------
    # Initial data errors
    # --------------------------------------------------------

    data = bernoulli_fault(

        (
            samples,
            3
        ),

        p_data
    )


    d0 = data[:, 0]
    d1 = data[:, 1]
    d2 = data[:, 2]


    syndrome1 = []
    syndrome2 = []


    # --------------------------------------------------------
    # Syndrome extraction
    # --------------------------------------------------------

    for _ in range(
        rounds
    ):

        a1 = torch.zeros_like(
            d0
        )

        a2 = torch.zeros_like(
            d0
        )


        # Z0 Z1 parity
        d0, a1 = noisy_syndrome_cnot(

            d0,
            a1,
            p_cnot
        )


        d1, a1 = noisy_syndrome_cnot(

            d1,
            a1,
            p_cnot
        )


        # Z1 Z2 parity
        d1, a2 = noisy_syndrome_cnot(

            d1,
            a2,
            p_cnot
        )


        d2, a2 = noisy_syndrome_cnot(

            d2,
            a2,
            p_cnot
        )


        # Measurement error
        a1 = torch.logical_xor(

            a1,

            bernoulli_fault(
                a1.shape,
                p_meas
            )
        )


        a2 = torch.logical_xor(

            a2,

            bernoulli_fault(
                a2.shape,
                p_meas
            )
        )


        syndrome1.append(
            a1
        )

        syndrome2.append(
            a2
        )


    # --------------------------------------------------------
    # Majority vote
    # --------------------------------------------------------

    s1 = torch.stack(
        syndrome1,
        dim=-1
    )


    s2 = torch.stack(
        syndrome2,
        dim=-1
    )


    threshold = (
        rounds // 2
        + 1
    )


    decoded_s1 = (

        s1
        .sum(dim=-1)

        >= threshold
    )


    decoded_s2 = (

        s2
        .sum(dim=-1)

        >= threshold
    )


    # --------------------------------------------------------
    # Decoder
    #
    # 10 -> q0
    # 11 -> q1
    # 01 -> q2
    # --------------------------------------------------------

    correct_q0 = torch.logical_and(

        decoded_s1,

        torch.logical_not(
            decoded_s2
        )
    )


    correct_q1 = torch.logical_and(

        decoded_s1,

        decoded_s2
    )


    correct_q2 = torch.logical_and(

        torch.logical_not(
            decoded_s1
        ),

        decoded_s2
    )


    # --------------------------------------------------------
    # Intended recovery
    # --------------------------------------------------------

    d0 = torch.logical_xor(
        d0,
        correct_q0
    )

    d1 = torch.logical_xor(
        d1,
        correct_q1
    )

    d2 = torch.logical_xor(
        d2,
        correct_q2
    )


    # --------------------------------------------------------
    # Recovery-gate failure
    #
    # Only matters if recovery was executed.
    # --------------------------------------------------------

    recovery_fault0 = torch.logical_and(

        correct_q0,

        bernoulli_fault(
            correct_q0.shape,
            p_recovery
        )
    )


    recovery_fault1 = torch.logical_and(

        correct_q1,

        bernoulli_fault(
            correct_q1.shape,
            p_recovery
        )
    )


    recovery_fault2 = torch.logical_and(

        correct_q2,

        bernoulli_fault(
            correct_q2.shape,
            p_recovery
        )
    )


    d0 = torch.logical_xor(
        d0,
        recovery_fault0
    )

    d1 = torch.logical_xor(
        d1,
        recovery_fault1
    )

    d2 = torch.logical_xor(
        d2,
        recovery_fault2
    )


    # --------------------------------------------------------
    # Logical majority failure
    # --------------------------------------------------------

    residual_count = (

        d0.long()

        +

        d1.long()

        +

        d2.long()
    )


    logical_failure = (
        residual_count >= 2
    )


    return logical_failure


# ============================================================
# ESTIMATE EFFECTIVE LOGICAL ERROR RATE
# ============================================================

def estimate_effective_error(
    p_data,
    p_cnot,
    p_meas,
    rounds,
    seed
):

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


    failures = (
        sample_qec_logical_failure(

            samples=BLOCK_SAMPLES,

            p_data=p_data,

            p_cnot=p_cnot,

            p_meas=p_meas,

            p_recovery=P_RECOVERY,

            rounds=rounds
        )
    )


    return (

        failures
        .float()
        .mean()
        .item()
    )


# ============================================================
# PART 1:
# BUILD DENSE CLASSIFIER CALIBRATION CURVE
# ============================================================

print("\n======================================")
print("BUILDING CLASSIFIER CALIBRATION CURVE")
print("======================================")

print(
    "Trajectories:",
    N_TRAJECTORIES
)

print(
    "Calibration seeds:",
    N_CALIBRATION_SEEDS
)

print(
    "Calibration points:",
    len(CALIBRATION_P)
)


calibration_accuracy = []


start_time = time.time()


for p_index, p in enumerate(
    CALIBRATION_P
):

    seed_accuracies = []


    for seed_index in range(
        N_CALIBRATION_SEEDS
    ):

        current_seed = (

            SEED

            +

            10000 * p_index

            +

            seed_index
        )


        accuracy = (
            evaluate_logical_error(

                float(p),

                current_seed
            )
        )


        seed_accuracies.append(
            accuracy
        )


    mean_accuracy = np.mean(
        seed_accuracies
    )


    calibration_accuracy.append(
        mean_accuracy
    )


    print(

        f"{p_index + 1:02d}/"
        f"{len(CALIBRATION_P)} | "

        f"p={p:.5f} | "

        f"accuracy="
        f"{mean_accuracy:.4f}"
    )


calibration_accuracy = np.array(
    calibration_accuracy
)


print(
    "\nCalibration time:",
    f"{time.time() - start_time:.1f}",
    "sec"
)


# ============================================================
# SAVE CALIBRATION
# ============================================================

CALIBRATION_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_logical_error_accuracy_calibration.csv"
)


with open(
    CALIBRATION_CSV,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([
        "logical_x_error_probability",
        "accuracy"
    ])


    for p, acc in zip(

        CALIBRATION_P,
        calibration_accuracy
    ):

        writer.writerow([
            p,
            acc
        ])


# ============================================================
# CALIBRATION FIGURE
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    CALIBRATION_P * 100,

    calibration_accuracy * 100,

    marker="o",

    markersize=3
)


plt.xlabel(
    "Effective Logical X-Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Classifier Sensitivity to Logical X Errors"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


CALIBRATION_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_logical_error_calibration.png"
)


plt.savefig(

    CALIBRATION_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# ACCURACY INTERPOLATION
# ============================================================

def accuracy_from_logical_error(
    p
):

    p = np.clip(

        p,

        CALIBRATION_P.min(),

        CALIBRATION_P.max()
    )


    return float(
        np.interp(

            p,

            CALIBRATION_P,

            calibration_accuracy
        )
    )


# ============================================================
# PART 2:
# QEC BREAK-EVEN GRID
# ============================================================

print("\n======================================")
print("QEC BREAK-EVEN SWEEP")
print("======================================")

print(
    "Block samples per point:",
    BLOCK_SAMPLES
)

print(
    "Recovery error probability:",
    P_RECOVERY
)


results = []


for data_index, p_data in enumerate(
    P_DATA_VALUES
):

    print(
        "\n======================================"
    )

    print(
        f"DATA ERROR p = "
        f"{p_data:.4f}"
    )

    print(
        "======================================"
    )


    for meas_index, p_meas in enumerate(
        P_MEAS_VALUES
    ):

        for cnot_index, p_cnot in enumerate(
            P_CNOT_VALUES
        ):


            base_seed = (

                SEED

                +

                1_000_000
                * data_index

                +

                10_000
                * meas_index

                +

                100
                * cnot_index
            )


            # =================================================
            # QEC-1 EFFECTIVE ERROR
            # =================================================

            p_eff_qec1 = (
                estimate_effective_error(

                    p_data=p_data,

                    p_cnot=p_cnot,

                    p_meas=p_meas,

                    rounds=1,

                    seed=base_seed
                )
            )


            # =================================================
            # QEC-3 EFFECTIVE ERROR
            # =================================================

            p_eff_qec3 = (
                estimate_effective_error(

                    p_data=p_data,

                    p_cnot=p_cnot,

                    p_meas=p_meas,

                    rounds=3,

                    seed=(
                        base_seed
                        + 50_000
                    )
                )
            )


            # =================================================
            # MAP LOGICAL ERROR -> CIFAR ACCURACY
            # =================================================

            accuracy_qec1 = (
                accuracy_from_logical_error(
                    p_eff_qec1
                )
            )


            accuracy_qec3 = (
                accuracy_from_logical_error(
                    p_eff_qec3
                )
            )


            delta_accuracy = (

                accuracy_qec3

                -

                accuracy_qec1
            )


            delta_error = (

                p_eff_qec1

                -

                p_eff_qec3
            )


            results.append({

                "p_data":
                    p_data,

                "p_cnot":
                    p_cnot,

                "p_meas":
                    p_meas,

                "p_recovery":
                    P_RECOVERY,

                "p_eff_qec1":
                    p_eff_qec1,

                "p_eff_qec3":
                    p_eff_qec3,

                "accuracy_qec1":
                    accuracy_qec1,

                "accuracy_qec3":
                    accuracy_qec3,

                "delta_accuracy":
                    delta_accuracy,

                "delta_error":
                    delta_error
            })


            print(

                f"pCNOT="
                f"{p_cnot:.3f} | "

                f"pMEAS="
                f"{p_meas:.3f} | "

                f"pL1="
                f"{p_eff_qec1:.5f} | "

                f"pL3="
                f"{p_eff_qec3:.5f} | "

                f"ΔA="
                f"{delta_accuracy * 100:+.2f} pp"
            )


# ============================================================
# SAVE GRID CSV
# ============================================================

GRID_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_qec_break_even_grid.csv"
)


with open(
    GRID_CSV,
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([

        "p_data",
        "p_cnot",
        "p_measurement",
        "p_recovery",

        "effective_qec1_error",

        "effective_qec3_error",

        "estimated_qec1_accuracy",

        "estimated_qec3_accuracy",

        "qec3_minus_qec1_accuracy",

        "qec1_minus_qec3_logical_error"
    ])


    for row in results:

        writer.writerow([

            row["p_data"],

            row["p_cnot"],

            row["p_meas"],

            row["p_recovery"],

            row["p_eff_qec1"],

            row["p_eff_qec3"],

            row["accuracy_qec1"],

            row["accuracy_qec3"],

            row["delta_accuracy"],

            row["delta_error"]
        ])


# ============================================================
# PART 3:
# HEATMAPS
#
# Positive:
#
# QEC-3 is better
#
# Negative:
#
# QEC-1 is better
# ============================================================

heatmap_paths = []


for p_data in P_DATA_VALUES:

    subset = [

        row

        for row in results

        if row["p_data"] == p_data
    ]


    gain_matrix = np.zeros(

        (
            len(P_MEAS_VALUES),
            len(P_CNOT_VALUES)
        )
    )


    logical_gain_matrix = np.zeros_like(
        gain_matrix
    )


    for row in subset:

        i = P_MEAS_VALUES.index(
            row["p_meas"]
        )

        j = P_CNOT_VALUES.index(
            row["p_cnot"]
        )


        gain_matrix[
            i,
            j
        ] = (

            row["delta_accuracy"]

            * 100
        )


        logical_gain_matrix[
            i,
            j
        ] = (

            row["delta_error"]

            * 100
        )


    # ========================================================
    # ACCURACY HEATMAP
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    image = plt.imshow(

        gain_matrix,

        origin="lower",

        aspect="auto"
    )


    plt.colorbar(
        image,
        label=(
            "QEC-3 − QEC-1 Accuracy "
            "(percentage points)"
        )
    )


    plt.xticks(

        np.arange(
            len(P_CNOT_VALUES)
        ),

        [
            f"{p * 100:.1f}"

            for p in P_CNOT_VALUES
        ]
    )


    plt.yticks(

        np.arange(
            len(P_MEAS_VALUES)
        ),

        [
            f"{p * 100:.1f}"

            for p in P_MEAS_VALUES
        ]
    )


    plt.xlabel(
        "Syndrome CNOT Error Probability (%)"
    )

    plt.ylabel(
        "Ancilla Measurement Error Probability (%)"
    )


    plt.title(

        "QEC-3 vs QEC-1 Classification Gain\n"

        f"Physical Data Error = "
        f"{p_data * 100:.1f}%"
    )


    # --------------------------------------------------------
    # Numerical annotations
    # --------------------------------------------------------

    for i in range(
        len(P_MEAS_VALUES)
    ):

        for j in range(
            len(P_CNOT_VALUES)
        ):

            plt.text(

                j,

                i,

                f"{gain_matrix[i, j]:+.1f}",

                ha="center",

                va="center",

                fontsize=8
            )


    # --------------------------------------------------------
    # Zero contour = break-even boundary
    # --------------------------------------------------------

    if (
        gain_matrix.min() < 0
        and
        gain_matrix.max() > 0
    ):

        plt.contour(

            gain_matrix,

            levels=[0],

            origin="lower",

            linewidths=2
        )


    plt.tight_layout()


    p_name = str(
        p_data
    ).replace(
        ".",
        "p"
    )


    heatmap_path = os.path.join(

        FIGURE_DIR,

        (
            "qec3_vs_qec1_accuracy_heatmap_"
            f"pdata_{p_name}.png"
        )
    )


    plt.savefig(

        heatmap_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    heatmap_paths.append(
        heatmap_path
    )


    # ========================================================
    # EFFECTIVE LOGICAL ERROR HEATMAP
    #
    # Positive:
    #
    # QEC-3 has LOWER logical error.
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    image = plt.imshow(

        logical_gain_matrix,

        origin="lower",

        aspect="auto"
    )


    plt.colorbar(

        image,

        label=(
            "QEC-1 − QEC-3 Logical Error "
            "(percentage points)"
        )
    )


    plt.xticks(

        np.arange(
            len(P_CNOT_VALUES)
        ),

        [
            f"{p * 100:.1f}"

            for p in P_CNOT_VALUES
        ]
    )


    plt.yticks(

        np.arange(
            len(P_MEAS_VALUES)
        ),

        [
            f"{p * 100:.1f}"

            for p in P_MEAS_VALUES
        ]
    )


    plt.xlabel(
        "Syndrome CNOT Error Probability (%)"
    )

    plt.ylabel(
        "Ancilla Measurement Error Probability (%)"
    )


    plt.title(

        "Logical-Error Benefit of Three Syndrome Rounds\n"

        f"Physical Data Error = "
        f"{p_data * 100:.1f}%"
    )


    for i in range(
        len(P_MEAS_VALUES)
    ):

        for j in range(
            len(P_CNOT_VALUES)
        ):

            plt.text(

                j,

                i,

                f"{logical_gain_matrix[i, j]:+.2f}",

                ha="center",

                va="center",

                fontsize=8
            )


    if (
        logical_gain_matrix.min() < 0
        and
        logical_gain_matrix.max() > 0
    ):

        plt.contour(

            logical_gain_matrix,

            levels=[0],

            origin="lower",

            linewidths=2
        )


    plt.tight_layout()


    logical_heatmap_path = os.path.join(

        FIGURE_DIR,

        (
            "qec3_vs_qec1_logical_error_heatmap_"
            f"pdata_{p_name}.png"
        )
    )


    plt.savefig(

        logical_heatmap_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    heatmap_paths.append(
        logical_heatmap_path
    )


# ============================================================
# SUMMARY:
# HOW MUCH OF GRID FAVORS QEC-3?
# ============================================================

print("\n======================================")
print("BREAK-EVEN SUMMARY")
print("======================================")


for p_data in P_DATA_VALUES:

    subset = [

        row

        for row in results

        if row["p_data"] == p_data
    ]


    qec3_better = sum(

        row["delta_accuracy"] > 0

        for row in subset
    )


    equal = sum(

        abs(
            row["delta_accuracy"]
        ) < 1e-8

        for row in subset
    )


    total = len(
        subset
    )


    print(

        f"p_data="
        f"{p_data:.3f} | "

        f"QEC-3 better in "
        f"{qec3_better}/{total} "
        f"grid points | "

        f"{100*qec3_better/total:.1f}%"
    )


# ============================================================
# FILE SUMMARY
# ============================================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Calibration CSV:",
    CALIBRATION_CSV
)

print(
    "Calibration figure:",
    CALIBRATION_FIGURE
)

print(
    "Break-even grid CSV:",
    GRID_CSV
)


for path in heatmap_paths:

    print(
        "Heatmap:",
        path
    )


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "Positive heatmap values mean "
    "QEC-3 gives higher estimated "
    "classification accuracy than QEC-1."
)

print(
    "Negative values mean the extra "
    "syndrome rounds introduce more damage "
    "than the additional syndrome information "
    "can correct."
)

print(
    "The zero contour is the approximate "
    "QEC-1 / QEC-3 break-even boundary."
)

print(
    "Classification values are estimated from "
    "a dense empirical accuracy-vs-logical-error "
    "calibration of the actual trained CIFAR-10 "
    "quantum classifier."
)

print(
    "The experiment still models only "
    "X / bit-flip errors."
)
