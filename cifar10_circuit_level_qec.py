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

# Monte-Carlo trajectories
N_TRAJECTORIES = 32

# Independent repetitions
N_SEEDS = 5


# ============================================================
# CIRCUIT-LEVEL QEC NOISE
# ============================================================

# Bit-flip probability associated with each
# syndrome-extraction CNOT participant.
P_CNOT = 0.005

# Ancilla measurement bit-flip probability.
P_MEAS = 0.010

# Bit-flip fault probability on an executed recovery X.
P_RECOVERY = 0.005


# ============================================================
# DATA/GATE ERROR SWEEP
# ============================================================

P_DATA_VALUES = [
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
# RANDOM SEEDS
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
        "CUDA is required for this experiment."
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
# LOAD DATA
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

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# MODEL
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
# RY
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
# CONDITIONAL X
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

        2.0
        * bit.float()
    )


    z_signs.append(
        signs
    )


Z_SIGN_MATRIX = torch.stack(
    z_signs,
    dim=1
)


# ============================================================
# BERNOULLI FAULT HELPER
# ============================================================

def bernoulli_fault(
    shape,
    probability
):

    if probability <= 0:

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

        < probability
    )


# ============================================================
# UNPROTECTED FAULT
# ============================================================

def sample_unprotected_fault(
    shape,
    p_data
):

    return bernoulli_fault(
        shape,
        p_data
    )


# ============================================================
# NOISY SYNDROME CNOT
#
# We track X errors only.
#
# Ideal CNOT:
#
# ancilla ^= data
#
# Then the noisy CNOT can introduce an X error on:
#
# - control/data
# - target/ancilla
#
# independently with probability P_CNOT.
# ============================================================

def noisy_syndrome_cnot(
    data_bit,
    ancilla_bit
):

    # Ideal CNOT propagation

    ancilla_bit = torch.logical_xor(
        ancilla_bit,
        data_bit
    )


    # CNOT-associated X fault on data/control

    data_bit = torch.logical_xor(

        data_bit,

        bernoulli_fault(
            data_bit.shape,
            P_CNOT
        )
    )


    # CNOT-associated X fault on ancilla/target

    ancilla_bit = torch.logical_xor(

        ancilla_bit,

        bernoulli_fault(
            ancilla_bit.shape,
            P_CNOT
        )
    )


    return (
        data_bit,
        ancilla_bit
    )


# ============================================================
# CIRCUIT-LEVEL REPETITION-CODE QEC
#
# Three physical data qubits:
#
# d0 d1 d2
#
#
# Syndrome ancillas measure:
#
# Z0 Z1 parity:
# s1 = d0 XOR d1
#
# Z1 Z2 parity:
# s2 = d1 XOR d2
#
#
# Extraction circuit per round:
#
# d0 -> a1
# d1 -> a1
#
# d1 -> a2
# d2 -> a2
#
# Four noisy CNOTs per syndrome round.
#
#
# rounds = 1:
# one extraction
#
# rounds = 3:
# three extractions +
# majority vote of syndrome bits.
#
#
# The physical data state persists between rounds,
# so additional rounds can themselves introduce faults.
# ============================================================

def sample_circuit_qec_fault(
    shape,
    p_data,
    rounds
):

    # ========================================================
    # INITIAL DATA ERRORS
    # ========================================================

    data_errors = bernoulli_fault(

        (
            *shape,
            3
        ),

        p_data
    )


    d0 = data_errors[
        ..., 0
    ]

    d1 = data_errors[
        ..., 1
    ]

    d2 = data_errors[
        ..., 2
    ]


    measured_s1_rounds = []
    measured_s2_rounds = []


    # ========================================================
    # SYNDROME ROUNDS
    # ========================================================

    for _ in range(
        rounds
    ):

        # Fresh ancillas initialized in |0>

        a1 = torch.zeros_like(
            d0
        )

        a2 = torch.zeros_like(
            d0
        )


        # ----------------------------------------------------
        # Measure parity d0 XOR d1
        # ----------------------------------------------------

        d0, a1 = noisy_syndrome_cnot(
            d0,
            a1
        )


        d1, a1 = noisy_syndrome_cnot(
            d1,
            a1
        )


        # ----------------------------------------------------
        # Measure parity d1 XOR d2
        # ----------------------------------------------------

        d1, a2 = noisy_syndrome_cnot(
            d1,
            a2
        )


        d2, a2 = noisy_syndrome_cnot(
            d2,
            a2
        )


        # ----------------------------------------------------
        # Noisy ancilla measurement
        # ----------------------------------------------------

        measured_s1 = torch.logical_xor(

            a1,

            bernoulli_fault(
                a1.shape,
                P_MEAS
            )
        )


        measured_s2 = torch.logical_xor(

            a2,

            bernoulli_fault(
                a2.shape,
                P_MEAS
            )
        )


        measured_s1_rounds.append(
            measured_s1
        )

        measured_s2_rounds.append(
            measured_s2
        )


    # ========================================================
    # STACK ROUNDS
    # ========================================================

    s1_stack = torch.stack(
        measured_s1_rounds,
        dim=-1
    )

    s2_stack = torch.stack(
        measured_s2_rounds,
        dim=-1
    )


    # ========================================================
    # MAJORITY VOTE
    #
    # rounds=1 -> threshold 1
    # rounds=3 -> threshold 2
    # ========================================================

    threshold = (
        rounds // 2
        + 1
    )


    decoded_s1 = (

        s1_stack
        .sum(
            dim=-1
        )

        >= threshold
    )


    decoded_s2 = (

        s2_stack
        .sum(
            dim=-1
        )

        >= threshold
    )


    # ========================================================
    # DECODER
    #
    # syndrome:
    #
    # 00 -> no correction
    # 10 -> q0
    # 11 -> q1
    # 01 -> q2
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
    # IDEAL RECOVERY ACTION
    # ========================================================

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


    # ========================================================
    # NOISY RECOVERY GATE
    #
    # A recovery X is executed only when decoder commands it.
    #
    # A faulty recovery contributes another X with
    # probability P_RECOVERY.
    #
    # X followed by X = identity, so this can cancel the
    # desired recovery.
    # ========================================================

    recovery_fault0 = torch.logical_and(

        correct_q0,

        bernoulli_fault(
            correct_q0.shape,
            P_RECOVERY
        )
    )


    recovery_fault1 = torch.logical_and(

        correct_q1,

        bernoulli_fault(
            correct_q1.shape,
            P_RECOVERY
        )
    )


    recovery_fault2 = torch.logical_and(

        correct_q2,

        bernoulli_fault(
            correct_q2.shape,
            P_RECOVERY
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


    # ========================================================
    # LOGICAL MAJORITY DECODING
    #
    # Two or three residual X errors => logical X.
    # ========================================================

    residual_errors = (

        d0.long()

        +

        d1.long()

        +

        d2.long()
    )


    logical_failure = (
        residual_errors >= 2
    )


    return logical_failure


# ============================================================
# SELECT ERROR MODEL
# ============================================================

def sample_faults(
    shape,
    p_data,
    mode
):

    if mode == "none":

        return sample_unprotected_fault(
            shape,
            p_data
        )


    if mode == "qec1":

        return sample_circuit_qec_fault(

            shape=shape,

            p_data=p_data,

            rounds=1
        )


    if mode == "qec3":

        return sample_circuit_qec_fault(

            shape=shape,

            p_data=p_data,

            rounds=3
        )


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# 8-QUBIT LOGICAL CIRCUIT
#
# The statevector still contains 8 logical qubits.
#
# For protected modes, the physical repetition-code block
# and syndrome circuit are Pauli-error tracked efficiently.
#
# The resulting logical X fault is then applied to the
# corresponding logical qubit.
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p_data,
    mode
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    # ========================================================
    # SAMPLE ALL LOGICAL FAULT EVENTS
    # ========================================================

    encoding_faults = sample_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p_data,

        mode
    )


    ry_faults = sample_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p_data,

        mode
    )


    rz_faults = sample_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        p_data,

        mode
    )


    cnot_faults = sample_faults(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2
        ),

        p_data,

        mode
    )


    # ========================================================
    # VQC
    # ========================================================

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


    # ========================================================
    # MEASURE <Z>
    # ========================================================

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
# ESTIMATE EFFECTIVE LOGICAL X ERROR
# ============================================================

def estimate_effective_error(
    p_data,
    mode,
    samples=500000
):

    if mode == "none":
        return p_data


    rounds = (
        1
        if mode == "qec1"
        else 3
    )


    logical_faults = (
        sample_circuit_qec_fault(

            shape=(samples,),

            p_data=p_data,

            rounds=rounds
        )
    )


    return (

        logical_faults
        .float()
        .mean()
        .item()
    )


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier(
    p_data,
    mode,
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


            # ------------------------------------------------
            # Ideal unprotected p=0 only requires one
            # trajectory.
            #
            # QEC at p_data=0 STILL uses trajectories because
            # the QEC circuitry itself is noisy.
            # ------------------------------------------------

            if (
                mode == "none"
                and p_data == 0
            ):

                trajectories = 1

            else:

                trajectories = (
                    N_TRAJECTORIES
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


            quantum_features = (
                run_quantum_circuit(

                    inputs=expanded_x,

                    quantum_weights=(
                        model.quantum_weights
                    ),

                    p_data=p_data,

                    mode=mode
                )
            )


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


(
    ideal_accuracy,
    ideal_f1,
    _
) = evaluate_classifier(

    p_data=0.0,

    mode="none",

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


# ============================================================
# EXPERIMENT
# ============================================================

print("\n======================================")
print("CIRCUIT-LEVEL QEC SETTINGS")
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
    "Syndrome CNOT p:",
    P_CNOT
)

print(
    "Measurement p:",
    P_MEAS
)

print(
    "Recovery p:",
    P_RECOVERY
)


detailed_results = []


for p_index, p_data in enumerate(
    P_DATA_VALUES
):

    print(
        "\n======================================"
    )

    print(
        f"p_data = "
        f"{p_data:.4f}"
    )

    print(
        "======================================"
    )


    # ========================================================
    # EFFECTIVE ERROR ESTIMATES
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
        f"Effective unprotected p = "
        f"{p_data:.6f}"
    )

    print(
        f"Effective QEC-1 p       = "
        f"{effective_qec1:.6f}"
    )

    print(
        f"Effective QEC-3 p       = "
        f"{effective_qec3:.6f}"
    )


    # ========================================================
    # MULTIPLE SEEDS
    # ========================================================

    for seed_index in range(
        N_SEEDS
    ):

        base_seed = (

            SEED

            +

            p_index * 1000

            +

            seed_index
        )


        # ----------------------------------------------------
        # UNPROTECTED
        # ----------------------------------------------------

        (
            acc_none,
            f1_none,
            time_none

        ) = evaluate_classifier(

            p_data=p_data,

            mode="none",

            random_seed=base_seed
        )


        # ----------------------------------------------------
        # ONE-ROUND CIRCUIT QEC
        # ----------------------------------------------------

        (
            acc_qec1,
            f1_qec1,
            time_qec1

        ) = evaluate_classifier(

            p_data=p_data,

            mode="qec1",

            random_seed=(
                base_seed
                + 100000
            )
        )


        # ----------------------------------------------------
        # THREE-ROUND CIRCUIT QEC
        # ----------------------------------------------------

        (
            acc_qec3,
            f1_qec3,
            time_qec3

        ) = evaluate_classifier(

            p_data=p_data,

            mode="qec3",

            random_seed=(
                base_seed
                + 200000
            )
        )


        detailed_results.append({

            "p_data":
                p_data,

            "seed":
                base_seed,

            "effective_qec1":
                effective_qec1,

            "effective_qec3":
                effective_qec3,

            "acc_none":
                acc_none,

            "acc_qec1":
                acc_qec1,

            "acc_qec3":
                acc_qec3,

            "f1_none":
                f1_none,

            "f1_qec1":
                f1_qec1,

            "f1_qec3":
                f1_qec3,

            "time_none":
                time_none,

            "time_qec1":
                time_qec1,

            "time_qec3":
                time_qec3
        })


        print(

            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "

            f"None="
            f"{acc_none:.4f} | "

            f"QEC-1="
            f"{acc_qec1:.4f} | "

            f"QEC-3="
            f"{acc_qec3:.4f}"
        )


# ============================================================
# DETAILED CSV
# ============================================================

DETAILED_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_circuit_level_qec_detailed.csv"
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

        "p_cnot",

        "p_measurement",

        "p_recovery",

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

            P_CNOT,

            P_MEAS,

            P_RECOVERY,

            row["seed"],

            row["effective_qec1"],

            row["effective_qec3"],

            row["acc_none"],

            row["acc_qec1"],

            row["acc_qec3"],

            row["f1_none"],

            row["f1_qec1"],

            row["f1_qec3"],

            row["time_none"],

            row["time_qec1"],

            row["time_qec3"]
        ])


# ============================================================
# SUMMARY
# ============================================================

summary = []


for p_data in P_DATA_VALUES:

    subset = [

        row

        for row in detailed_results

        if row["p_data"]
        == p_data
    ]


    none = np.array([

        row["acc_none"]

        for row in subset
    ])


    qec1 = np.array([

        row["acc_qec1"]

        for row in subset
    ])


    qec3 = np.array([

        row["acc_qec3"]

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

        "none_mean":
            none.mean(),

        "none_std":
            none.std(ddof=1),

        "qec1_mean":
            qec1.mean(),

        "qec1_std":
            qec1.std(ddof=1),

        "qec3_mean":
            qec3.mean(),

        "qec3_std":
            qec3.std(ddof=1)
    })


# ============================================================
# SUMMARY CSV
# ============================================================

SUMMARY_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_circuit_level_qec_summary.csv"
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

        "p_cnot",

        "p_measurement",

        "p_recovery",

        "effective_unprotected_p",

        "effective_qec1_p",

        "effective_qec3_p",

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

            P_CNOT,

            P_MEAS,

            P_RECOVERY,

            row["p_data"],

            row["effective_qec1"],

            row["effective_qec3"],

            row["none_mean"],

            row["none_std"],

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

        f"{row['none_mean']:.4f}"
        f" ± "
        f"{row['none_std']:.4f}\t"

        f"{row['qec1_mean']:.4f}"
        f" ± "
        f"{row['qec1_std']:.4f}\t"

        f"{row['qec3_mean']:.4f}"
        f" ± "
        f"{row['qec3_std']:.4f}"
    )


print("\n======================================")
print("EFFECTIVE LOGICAL ERROR")
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

        f"{row['p_data']:.6f}\t"

        f"{row['effective_qec1']:.6f}\t"

        f"{row['effective_qec3']:.6f}"
    )


# ============================================================
# ARRAYS
# ============================================================

p_values = np.array([

    row["p_data"]

    for row in summary
])


none_mean = np.array([

    row["none_mean"]

    for row in summary
])


none_std = np.array([

    row["none_std"]

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

    none_mean * 100,

    marker="o",

    linewidth=2,

    label="Unprotected"
)


plt.fill_between(

    p_values * 100,

    (
        none_mean
        -
        none_std
    ) * 100,

    (
        none_mean
        +
        none_std
    ) * 100,

    alpha=0.15
)


plt.plot(

    p_values * 100,

    qec1_mean * 100,

    marker="s",

    linewidth=2,

    label="1-round circuit-level QEC"
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

    label="3-round circuit-level QEC"
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
    "Circuit-Level Noisy QEC for Quantum CIFAR-10"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_circuit_level_qec_accuracy.png"
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

    label="QEC-1"
)


plt.plot(

    p_values * 100,

    effective_qec3 * 100,

    marker="^",

    linewidth=2,

    label="QEC-3"
)


plt.xlabel(
    "Physical Data Error Probability (%)"
)

plt.ylabel(
    "Effective Logical X-Error Probability (%)"
)

plt.title(
    "Logical Error with Noisy QEC Circuitry"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ERROR_FIGURE = os.path.join(

    FIGURE_DIR,

    "circuit_level_qec_logical_error.png"
)


plt.savefig(

    ERROR_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
# QEC-3 VS QEC-1
# ============================================================

gain = (
    qec3_mean
    -
    qec1_mean
)


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    gain * 100,

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
    "QEC-3 − QEC-1 Accuracy (percentage points)"
)

plt.title(
    "Benefit or Cost of Repeated Syndrome Extraction"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GAIN_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_circuit_qec_three_round_gain.png"
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
    "Accuracy:",
    ACCURACY_FIGURE
)

print(
    "Logical error:",
    ERROR_FIGURE
)

print(
    "QEC-3 gain:",
    GAIN_FIGURE
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "Syndrome extraction CNOTs are noisy."
)

print(
    "Ancilla measurements are noisy."
)

print(
    "Recovery X gates are noisy."
)

print(
    "Repeated syndrome rounds can therefore "
    "both help and introduce additional faults."
)

print(
    "The repetition code still protects only "
    "against X / bit-flip errors."
)

print(
    "The protected VQC is represented using "
    "Pauli-error tracking rather than an explicit "
    "24+ qubit encoded statevector."
)

print(
    "This is therefore a circuit-level "
    "repetition-code FT proxy, not yet a general "
    "fault-tolerant implementation of RY/RZ gates."
)