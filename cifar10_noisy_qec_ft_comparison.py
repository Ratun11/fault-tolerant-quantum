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

# Monte-Carlo quantum-noise trajectories
N_TRAJECTORIES = 32

# Independent repetitions
N_SEEDS = 5

# Syndrome-bit measurement error probability
P_SYNDROME = 0.01


# Physical data/gate bit-flip probability
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

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# TRAINED QUANTUM CLASSIFIER
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
# QUANTUM STATE
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

    a = state_view[
        :, :, 0, :
    ]

    b = state_view[
        :, :, 1, :
    ]


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


    a = state_view[
        :, :, 0, :
    ]

    b = state_view[
        :, :, 1, :
    ]


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


    out0 = (
        phase0 * a
    )

    out1 = (
        phase1 * b
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


    state_view = state.reshape(
        batch_size,
        left,
        2,
        right
    )


    a = state_view[
        :, :, 0, :
    ]

    b = state_view[
        :, :, 1, :
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
# Z EXPECTATION VALUES
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
# SIMPLE UNPROTECTED X FAULT
# ============================================================

def sample_unprotected_fault(
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
# QEC LOGICAL-FAILURE SAMPLER
#
# Three physical qubits represent one logical qubit.
#
#
# Physical errors:
#
# e0 e1 e2
#
#
# Syndrome:
#
# s1 = e0 XOR e1
#
# s2 = e1 XOR e2
#
#
# Decoder:
#
# 00 -> no correction
#
# 10 -> correct q0
#
# 11 -> correct q1
#
# 01 -> correct q2
#
#
# After recovery, logical failure occurs if >=2 physical
# errors remain.
#
#
# Syndrome measurements themselves are noisy.
# ============================================================

def sample_qec_logical_fault(
    shape,
    p_data,
    p_syndrome,
    syndrome_rounds
):

    # ========================================================
    # THREE PHYSICAL DATA QUBITS
    # ========================================================

    if p_data <= 0:

        data_errors = torch.zeros(

            *shape,
            3,

            dtype=torch.bool,

            device=device
        )

    else:

        data_errors = (

            torch.rand(

                *shape,
                3,

                device=device
            )

            < p_data
        )


    e0 = data_errors[
        ..., 0
    ]

    e1 = data_errors[
        ..., 1
    ]

    e2 = data_errors[
        ..., 2
    ]


    # ========================================================
    # TRUE SYNDROME
    # ========================================================

    true_s1 = torch.logical_xor(
        e0,
        e1
    )

    true_s2 = torch.logical_xor(
        e1,
        e2
    )


    # ========================================================
    # REPEATED SYNDROME EXTRACTION
    # ========================================================

    syndrome_shape = (
        *shape,
        syndrome_rounds
    )


    if p_syndrome <= 0:

        syndrome_noise_1 = torch.zeros(

            syndrome_shape,

            dtype=torch.bool,

            device=device
        )

        syndrome_noise_2 = torch.zeros(

            syndrome_shape,

            dtype=torch.bool,

            device=device
        )

    else:

        syndrome_noise_1 = (

            torch.rand(

                syndrome_shape,

                device=device
            )

            < p_syndrome
        )


        syndrome_noise_2 = (

            torch.rand(

                syndrome_shape,

                device=device
            )

            < p_syndrome
        )


    measured_s1 = torch.logical_xor(

        true_s1.unsqueeze(-1),

        syndrome_noise_1
    )


    measured_s2 = torch.logical_xor(

        true_s2.unsqueeze(-1),

        syndrome_noise_2
    )


    # ========================================================
    # MAJORITY VOTE
    #
    # For one round, this just returns that measurement.
    #
    # For three rounds, at least two must be 1.
    # ========================================================

    threshold = (
        syndrome_rounds // 2
        + 1
    )


    decoded_s1 = (

        measured_s1
        .sum(
            dim=-1
        )

        >= threshold
    )


    decoded_s2 = (

        measured_s2
        .sum(
            dim=-1
        )

        >= threshold
    )


    # ========================================================
    # RECOVERY DECISION
    # ========================================================

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


    # ========================================================
    # APPLY RECOVERY
    #
    # Recovery gates themselves are ideal in this experiment.
    # ========================================================

    residual_e0 = torch.logical_xor(
        e0,
        correct_q0
    )

    residual_e1 = torch.logical_xor(
        e1,
        correct_q1
    )

    residual_e2 = torch.logical_xor(
        e2,
        correct_q2
    )


    # ========================================================
    # MAJORITY DECODE
    #
    # >= 2 residual X errors means logical X failure.
    # ========================================================

    residual_count = (

        residual_e0.long()

        +

        residual_e1.long()

        +

        residual_e2.long()
    )


    logical_failure = (
        residual_count >= 2
    )


    return logical_failure


# ============================================================
# QUANTUM CIRCUIT
#
# protection_mode:
#
# "none"
# "qec1"
# "qec3"
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p_data,
    protection_mode
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    # ========================================================
    # CHOOSE FAULT MODEL
    # ========================================================

    def get_faults(
        shape
    ):

        if protection_mode == "none":

            return sample_unprotected_fault(
                shape,
                p_data
            )


        elif protection_mode == "qec1":

            return sample_qec_logical_fault(

                shape=shape,

                p_data=p_data,

                p_syndrome=P_SYNDROME,

                syndrome_rounds=1
            )


        elif protection_mode == "qec3":

            return sample_qec_logical_fault(

                shape=shape,

                p_data=p_data,

                p_syndrome=P_SYNDROME,

                syndrome_rounds=3
            )


        else:

            raise ValueError(
                "Unknown protection mode."
            )


    # ========================================================
    # FAULT MASKS
    # ========================================================

    encoding_faults = get_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        )
    )


    ry_faults = get_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        )
    )


    rz_faults = get_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        )
    )


    cnot_faults = get_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2
        )
    )


    # ========================================================
    # QUANTUM CIRCUIT
    # ========================================================

    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            * N_QUBITS
        )


        # ----------------------------------------------------
        # DATA ENCODING
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
        # TRAINABLE ROTATIONS
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
        # CNOT RING
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


    # ========================================================
    # EXPECTATION VALUES
    # ========================================================

    probabilities = (

        state.real.square()

        +

        state.imag.square()
    )


    quantum_features = (

        probabilities

        @

        Z_SIGN_MATRIX
    )


    return quantum_features


# ============================================================
# ESTIMATE EFFECTIVE LOGICAL ERROR RATE
#
# This does not run the QML model.
# It directly samples many repetition-code blocks so we can
# inspect how much 1-round and 3-round QEC suppress X errors.
# ============================================================

def estimate_effective_error(
    p_data,
    mode,
    samples=500000
):

    if mode == "none":

        return p_data


    if mode == "qec1":

        rounds = 1

    elif mode == "qec3":

        rounds = 3

    else:

        raise ValueError(
            "Unknown mode"
        )


    faults = sample_qec_logical_fault(

        shape=(samples,),

        p_data=p_data,

        p_syndrome=P_SYNDROME,

        syndrome_rounds=rounds
    )


    return (
        faults.float()
        .mean()
        .item()
    )


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier(
    p_data,
    protection_mode,
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

    start_time = time.time()


    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(
                device,
                non_blocking=True
            )


            batch_size = (
                x.shape[0]
            )


            if p_data == 0:

                trajectories = 1

            else:

                trajectories = (
                    N_TRAJECTORIES
                )


            # =================================================
            # REPEAT FOR NOISE TRAJECTORIES
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
                    batch_size
                    * trajectories,
                    32
                )
            )


            # =================================================
            # QUANTUM SIMULATION
            # =================================================

            quantum_features = (
                run_quantum_circuit(

                    inputs=expanded_x,

                    quantum_weights=(
                        model.quantum_weights
                    ),

                    p_data=p_data,

                    protection_mode=(
                        protection_mode
                    )
                )
            )


            # =================================================
            # TRAJECTORY AVERAGING
            # =================================================

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


            # =================================================
            # CLASSICAL HEAD
            # =================================================

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


ideal_accuracy, ideal_f1, _ = (
    evaluate_classifier(

        p_data=0.0,

        protection_mode="none",

        random_seed=SEED
    )
)


print(
    f"Ideal accuracy = "
    f"{ideal_accuracy:.4f}"
)

print(
    f"Ideal Macro-F1 = "
    f"{ideal_f1:.4f}"
)


# ============================================================
# EXPERIMENT
# ============================================================

detailed_results = []


print("\n======================================")
print("NOISY QEC / FT-STYLE EXPERIMENT")
print("======================================")

print(
    "Trajectories:",
    N_TRAJECTORIES
)

print(
    "Seeds:",
    N_SEEDS
)

print(
    "Syndrome error probability:",
    P_SYNDROME
)


for p_index, p_data in enumerate(
    NOISE_VALUES
):

    print(
        "\n======================================"
    )

    print(
        f"Physical data error p = "
        f"{p_data:.4f}"
    )

    print(
        "======================================"
    )


    # ========================================================
    # ESTIMATE EFFECTIVE PER-LOCATION ERROR
    # ========================================================

    torch.manual_seed(
        SEED + p_index
    )

    torch.cuda.manual_seed_all(
        SEED + p_index
    )


    effective_qec1 = (
        estimate_effective_error(

            p_data,

            "qec1"
        )
    )


    effective_qec3 = (
        estimate_effective_error(

            p_data,

            "qec3"
        )
    )


    print(
        f"Unprotected effective p = "
        f"{p_data:.6f}"
    )

    print(
        f"1-round QEC effective p = "
        f"{effective_qec1:.6f}"
    )

    print(
        f"3-round QEC effective p = "
        f"{effective_qec3:.6f}"
    )


    # ========================================================
    # MULTIPLE RANDOM SEEDS
    # ========================================================

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


        # ----------------------------------------------------
        # p = 0 can reuse ideal result
        # ----------------------------------------------------

        if p_data == 0:

            unprotected_acc = (
                ideal_accuracy
            )

            unprotected_f1 = (
                ideal_f1
            )

            qec1_acc = (
                ideal_accuracy
            )

            qec1_f1 = (
                ideal_f1
            )

            qec3_acc = (
                ideal_accuracy
            )

            qec3_f1 = (
                ideal_f1
            )

            t_unprotected = 0.0
            t_qec1 = 0.0
            t_qec3 = 0.0


        else:

            # =================================================
            # UNPROTECTED
            # =================================================

            (
                unprotected_acc,
                unprotected_f1,
                t_unprotected

            ) = evaluate_classifier(

                p_data=p_data,

                protection_mode="none",

                random_seed=current_seed
            )


            # =================================================
            # ONE-ROUND QEC
            # =================================================

            (
                qec1_acc,
                qec1_f1,
                t_qec1

            ) = evaluate_classifier(

                p_data=p_data,

                protection_mode="qec1",

                random_seed=(
                    current_seed
                    + 100000
                )
            )


            # =================================================
            # THREE-ROUND QEC
            # =================================================

            (
                qec3_acc,
                qec3_f1,
                t_qec3

            ) = evaluate_classifier(

                p_data=p_data,

                protection_mode="qec3",

                random_seed=(
                    current_seed
                    + 200000
                )
            )


        detailed_results.append({

            "p_data":
                p_data,

            "p_syndrome":
                P_SYNDROME,

            "seed":
                current_seed,

            "effective_qec1":
                effective_qec1,

            "effective_qec3":
                effective_qec3,

            "unprotected_accuracy":
                unprotected_acc,

            "qec1_accuracy":
                qec1_acc,

            "qec3_accuracy":
                qec3_acc,

            "unprotected_f1":
                unprotected_f1,

            "qec1_f1":
                qec1_f1,

            "qec3_f1":
                qec3_f1,

            "unprotected_seconds":
                t_unprotected,

            "qec1_seconds":
                t_qec1,

            "qec3_seconds":
                t_qec3
        })


        print(

            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "

            f"None="
            f"{unprotected_acc:.4f} | "

            f"QEC-1="
            f"{qec1_acc:.4f} | "

            f"QEC-3="
            f"{qec3_acc:.4f}"
        )


# ============================================================
# SAVE DETAILED CSV
# ============================================================

DETAILED_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_noisy_qec_ft_detailed.csv"
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

        "p_data",
        "p_syndrome",
        "seed",

        "effective_qec1",
        "effective_qec3",

        "unprotected_accuracy",
        "qec1_accuracy",
        "qec3_accuracy",

        "unprotected_macro_f1",
        "qec1_macro_f1",
        "qec3_macro_f1",

        "unprotected_seconds",
        "qec1_seconds",
        "qec3_seconds"
    ])


    for row in detailed_results:

        writer.writerow([

            row["p_data"],
            row["p_syndrome"],
            row["seed"],

            row["effective_qec1"],
            row["effective_qec3"],

            row["unprotected_accuracy"],
            row["qec1_accuracy"],
            row["qec3_accuracy"],

            row["unprotected_f1"],
            row["qec1_f1"],
            row["qec3_f1"],

            row["unprotected_seconds"],
            row["qec1_seconds"],
            row["qec3_seconds"]
        ])


# ============================================================
# SUMMARY
# ============================================================

summary = []


for p_data in NOISE_VALUES:

    subset = [

        row

        for row in detailed_results

        if row["p_data"] == p_data
    ]


    unprotected = np.array([

        row["unprotected_accuracy"]

        for row in subset
    ])


    qec1 = np.array([

        row["qec1_accuracy"]

        for row in subset
    ])


    qec3 = np.array([

        row["qec3_accuracy"]

        for row in subset
    ])


    summary.append({

        "p_data":
            p_data,

        "effective_qec1":
            subset[0][
                "effective_qec1"
            ],

        "effective_qec3":
            subset[0][
                "effective_qec3"
            ],

        "unprotected_mean":
            unprotected.mean(),

        "unprotected_std":
            unprotected.std(ddof=1)
            if len(unprotected) > 1
            else 0.0,

        "qec1_mean":
            qec1.mean(),

        "qec1_std":
            qec1.std(ddof=1)
            if len(qec1) > 1
            else 0.0,

        "qec3_mean":
            qec3.mean(),

        "qec3_std":
            qec3.std(ddof=1)
            if len(qec3) > 1
            else 0.0
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

SUMMARY_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_noisy_qec_ft_summary.csv"
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

        "p_data",

        "p_syndrome",

        "effective_error_unprotected",

        "effective_error_qec1",

        "effective_error_qec3",

        "unprotected_accuracy_mean",

        "unprotected_accuracy_std",

        "qec1_accuracy_mean",

        "qec1_accuracy_std",

        "qec3_accuracy_mean",

        "qec3_accuracy_std"
    ])


    for row in summary:

        writer.writerow([

            row["p_data"],

            P_SYNDROME,

            row["p_data"],

            row["effective_qec1"],

            row["effective_qec3"],

            row["unprotected_mean"],

            row["unprotected_std"],

            row["qec1_mean"],

            row["qec1_std"],

            row["qec3_mean"],

            row["qec3_std"]
        ])


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p_data\t\t"
    "Unprotected\t"
    "QEC-1\t\t"
    "QEC-3"
)


for row in summary:

    print(

        f"{row['p_data']:.4f}\t\t"

        f"{row['unprotected_mean']:.4f}"
        f" ± "
        f"{row['unprotected_std']:.4f}\t"

        f"{row['qec1_mean']:.4f}"
        f" ± "
        f"{row['qec1_std']:.4f}\t"

        f"{row['qec3_mean']:.4f}"
        f" ± "
        f"{row['qec3_std']:.4f}"
    )


# ============================================================
# PLOT ARRAYS
# ============================================================

p_values = np.array([

    row["p_data"]

    for row in summary
])


unprotected_mean = np.array([

    row["unprotected_mean"]

    for row in summary
])


unprotected_std = np.array([

    row["unprotected_std"]

    for row in summary
])


qec1_mean = np.array([

    row["qec1_mean"]

    for row in summary
])


qec1_std = np.array([

    row["qec1_std"]

    for row in summary
])


qec3_mean = np.array([

    row["qec3_mean"]

    for row in summary
])


qec3_std = np.array([

    row["qec3_std"]

    for row in summary
])


effective_qec1 = np.array([

    row["effective_qec1"]

    for row in summary
])


effective_qec3 = np.array([

    row["effective_qec3"]

    for row in summary
])


# ============================================================
# FIGURE 1
# CLASSIFICATION ACCURACY
# ============================================================

plt.figure(
    figsize=(8.5, 6)
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

    label="Unprotected"
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

    alpha=0.15
)


plt.plot(

    p_values * 100,

    qec1_mean * 100,

    marker="s",

    linewidth=2,

    label=(
        "1-round noisy QEC"
    )
)


plt.fill_between(

    p_values * 100,

    (
        qec1_mean
        -
        qec1_std
    ) * 100,

    (
        qec1_mean
        +
        qec1_std
    ) * 100,

    alpha=0.15
)


plt.plot(

    p_values * 100,

    qec3_mean * 100,

    marker="^",

    linewidth=2,

    label=(
        "3-round syndrome-voted QEC"
    )
)


plt.fill_between(

    p_values * 100,

    (
        qec3_mean
        -
        qec3_std
    ) * 100,

    (
        qec3_mean
        +
        qec3_std
    ) * 100,

    alpha=0.15
)


plt.xlabel(
    "Physical Data Bit-Flip Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Noisy QEC and Repeated Syndrome Extraction"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_noisy_qec_ft_accuracy.png"
)


plt.savefig(

    ACCURACY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2
# EFFECTIVE LOGICAL ERROR
# ============================================================

plt.figure(
    figsize=(8, 5.8)
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

    effective_qec1 * 100,

    marker="s",

    linewidth=2,

    label="1-round noisy QEC"
)


plt.plot(

    p_values * 100,

    effective_qec3 * 100,

    marker="^",

    linewidth=2,

    label="3-round syndrome-voted QEC"
)


plt.xlabel(
    "Physical Data Error Probability (%)"
)

plt.ylabel(
    "Effective Logical X-Error Probability (%)"
)

plt.title(
    "Logical Error Suppression with Noisy Syndromes"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ERROR_FIGURE = os.path.join(

    FIGURE_DIR,

    "noisy_qec_effective_logical_error.png"
)


plt.savefig(

    ERROR_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
# BENEFIT OF THREE-ROUND OVER ONE-ROUND QEC
# ============================================================

ft_gain = (
    qec3_mean
    -
    qec1_mean
)


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    ft_gain * 100,

    marker="o",

    linewidth=2
)


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Physical Data Error Probability (%)"
)

plt.ylabel(
    "Accuracy Gain (percentage points)"
)

plt.title(
    "Benefit of Repeated Syndrome Extraction"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GAIN_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_three_round_qec_gain.png"
)


plt.savefig(

    GAIN_FIGURE,

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
    "Accuracy figure:",
    ACCURACY_FIGURE
)

print(
    "Logical-error figure:",
    ERROR_FIGURE
)

print(
    "3-round gain figure:",
    GAIN_FIGURE
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "QEC-1 uses one noisy syndrome measurement."
)

print(
    "QEC-3 repeats syndrome extraction three times "
    "and majority-votes each syndrome bit."
)

print(
    "The syndrome-bit error probability is:",
    P_SYNDROME
)

print(
    "Recovery gates are still assumed perfect."
)

print(
    "Only X / bit-flip errors are modeled."
)

print(
    "Therefore QEC-3 is FT-style repeated syndrome "
    "extraction, not yet a complete fault-tolerant "
    "quantum computer."
)