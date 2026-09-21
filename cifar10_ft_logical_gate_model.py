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
# GATE-AWARE FAULT-TOLERANT LOGICAL-CIRCUIT MODEL
#
# Stage after:
#   cifar10_5qubit_syndrome_history.py
#
# Why this stage exists
# ---------------------
# Until now, the CIFAR-10 experiments effectively applied the
# same logical Pauli channel at every modeled fault location.
#
# A real fault-tolerant implementation does NOT give every
# logical operation the same cost:
#
#   * single-block logical Clifford operations can be
#     relatively cheap for the [[5,1,3]] code;
#
#   * arbitrary logical RY/RZ rotations are non-Clifford and
#     require a non-transversal/injection/synthesis gadget;
#
#   * a logical entangling gate such as CNOT requires a
#     pieceable fault-tolerant construction and intermediate
#     error correction.
#
# This script therefore introduces GATE-SPECIFIC logical
# channels.
#
# IMPORTANT SCIENTIFIC LIMITATION
# -------------------------------
# This is a logical-gadget / resource-aware proxy.
#
# It does NOT claim that the arbitrary RY/RZ rotations below
# have been explicitly synthesized into a complete physical
# Clifford+T circuit, and it does NOT explicitly simulate the
# full joint [[10,2,3]] intermediate decoder of the published
# pieceable five-qubit-code CNOT construction.
#
# Instead, it answers the next research question:
#
#   "What happens when non-Clifford rotations and entangling
#    gates are charged MORE fault-tolerant EC overhead than
#    the one-channel-per-location approximation?"
#
# The best syndrome-history logical error rate from the
# previous experiment is used as the base EC-protected
# logical Pauli channel.
#
# Because the previous CSV stores total p_L but not separate
# X/Y/Z probabilities, the residual channel is symmetrized:
#
#   [1-p_L, p_L/3, p_L/3, p_L/3].
#
# This assumption is printed and saved in the results.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

N_QUBITS = 8
N_CLASSES = 10
N_UPLOADS = 4

BATCH_SIZE = 512

N_TRAJECTORIES = 32
N_SEEDS = 5


# ============================================================
# LOGICAL GADGET OVERHEAD MODEL
# ============================================================
#
# One "cycle" means one application of the syndrome-history
# protected logical Pauli channel estimated previously.
#
# Uniform baseline:
#     one channel at every logical fault location.
#
# Gate-aware model:
#
#   DATA RY:
#       arbitrary logical rotation/injection proxy
#
#   TRAINABLE RY:
#       arbitrary logical rotation/injection proxy
#
#   TRAINABLE RZ:
#       arbitrary logical rotation/injection proxy
#
#   CNOT:
#       pieceable logical entangling-gate proxy
#
# The default values below are deliberately modest.
# They are NOT presented as exact physical gate counts.
# ============================================================

ROTATION_GADGET_CYCLES = 2

CNOT_GADGET_CYCLES = 2


# Also evaluate sensitivity to non-Clifford rotation overhead.
ROTATION_CYCLE_SWEEP = [
    1,
    2,
    3,
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

HISTORY_CHANNEL_CSV = (
    "results/"
    "five_qubit_syndrome_history_channels.csv"
)

RESULT_DIR = "results"
FIGURE_DIR = "figures"

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required for this experiment."
    )


device = torch.device(
    "cuda"
)


print("\n======================================")
print("DEVICE")
print("======================================")

print(
    "GPU:",
    torch.cuda.get_device_name(0),
)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA:",
    torch.version.cuda,
)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True,
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
    ),
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


print("\n======================================")
print("TEST DATA")
print("======================================")

print(
    "X_test:",
    X_test.shape,
)

print(
    "y_test:",
    y_test.shape,
)


# ============================================================
# TRAINED HYBRID QUANTUM CLASSIFIER
# ============================================================

class QuantumClassifier(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        self.quantum_weights = nn.Parameter(
            0.05
            *
            torch.randn(
                N_UPLOADS,
                N_QUBITS,
                2,
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                N_QUBITS,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(
                0.10
            ),
            nn.Linear(
                32,
                N_CLASSES,
            ),
        )


model = QuantumClassifier()


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
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
    MODEL_PATH,
)


# ============================================================
# LOAD SYNDROME-HISTORY LOGICAL ERROR RATES
# ============================================================

if not os.path.exists(
    HISTORY_CHANNEL_CSV
):

    raise FileNotFoundError(
        "\nRequired file not found:\n"
        f"{HISTORY_CHANNEL_CSV}\n\n"
        "Run cifar10_5qubit_syndrome_history.py first."
    )


HISTORY_LOGICAL_ERROR = {}


with open(
    HISTORY_CHANNEL_CSV,
    "r",
) as file:

    reader = csv.DictReader(
        file
    )


    for row in reader:

        p = float(
            row[
                "p"
            ]
        )

        p_logical = float(
            row[
                "history3_logical_error"
            ]
        )

        HISTORY_LOGICAL_ERROR[
            p
        ] = p_logical


P_VALUES = sorted(
    HISTORY_LOGICAL_ERROR.keys()
)


print("\n======================================")
print("SYNDROME-HISTORY CHANNEL")
print("======================================")

print(
    "Residual channel approximation:"
)

print(
    "I = 1-pL, X = Y = Z = pL/3"
)


for p in P_VALUES:

    print(
        f"p={p:.4f} | "
        f"history pL="
        f"{HISTORY_LOGICAL_ERROR[p]:.6f}"
    )


# ============================================================
# ONE-QUBIT PAULI CHANNEL UTILITIES
#
# Index:
# 0 I
# 1 X
# 2 Y
# 3 Z
#
# Phases are irrelevant for a Pauli channel.
# ============================================================

PAULI_BITS = [
    (0, 0),  # I
    (1, 0),  # X
    (1, 1),  # Y
    (0, 1),  # Z
]


BITS_TO_PAULI = {
    bits: index
    for index, bits in enumerate(
        PAULI_BITS
    )
}


def symmetric_pauli_channel(
    p_logical,
):

    p_logical = float(
        np.clip(
            p_logical,
            0.0,
            1.0,
        )
    )


    return np.array(
        [
            1.0
            -
            p_logical,

            p_logical
            /
            3.0,

            p_logical
            /
            3.0,

            p_logical
            /
            3.0,
        ],
        dtype=np.float64,
    )


def compose_pauli_channels(
    first,
    second,
):

    """
    Channel composition:
        apply first, then second.

    For Pauli channels the resulting Pauli label is the
    symplectic XOR of the two labels, up to a global phase.
    """

    output = np.zeros(
        4,
        dtype=np.float64,
    )


    for i in range(
        4
    ):

        xi, zi = (
            PAULI_BITS[
                i
            ]
        )


        for j in range(
            4
        ):

            xj, zj = (
                PAULI_BITS[
                    j
                ]
            )


            result_bits = (
                xi ^ xj,
                zi ^ zj,
            )


            k = (
                BITS_TO_PAULI[
                    result_bits
                ]
            )


            output[
                k
            ] += (
                first[
                    i
                ]
                *
                second[
                    j
                ]
            )


    output = (
        output
        /
        output.sum()
    )


    return output


def channel_power(
    channel,
    n,
):

    if n < 0:

        raise ValueError(
            "Channel power must be non-negative."
        )


    output = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


    for _ in range(
        n
    ):

        output = (
            compose_pauli_channels(
                output,
                channel,
            )
        )


    return output


# ============================================================
# BUILD GATE-SPECIFIC CHANNELS
# ============================================================

def base_history_channel(
    p,
):

    return (
        symmetric_pauli_channel(
            HISTORY_LOGICAL_ERROR[
                p
            ]
        )
    )


def unprotected_channel(
    p,
):

    return np.array(
        [
            1.0
            -
            p,

            p
            /
            3.0,

            p
            /
            3.0,

            p
            /
            3.0,
        ],
        dtype=np.float64,
    )


def gate_channels(
    p,
    mode,
    rotation_cycles=None,
):

    """
    Returns:
        encoding_rotation_channel
        trainable_rotation_channel
        cnot_participant_channel
    """


    if mode == "unprotected":

        physical = (
            unprotected_channel(
                p
            )
        )

        return (
            physical,
            physical,
            physical,
        )


    base = (
        base_history_channel(
            p
        )
    )


    if mode == "uniform_history":

        return (
            base,
            base,
            base,
        )


    if mode == "gate_aware_ft":

        if rotation_cycles is None:

            rotation_cycles = (
                ROTATION_GADGET_CYCLES
            )


        rotation_channel = (
            channel_power(
                base,
                rotation_cycles,
            )
        )


        cnot_channel = (
            channel_power(
                base,
                CNOT_GADGET_CYCLES,
            )
        )


        return (
            rotation_channel,
            rotation_channel,
            cnot_channel,
        )


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# PRINT LOGICAL-GATE ERROR TABLE
# ============================================================

print("\n======================================")
print("GATE-AWARE LOGICAL ERROR RATES")
print("======================================")

print(
    "Rotation gadget cycles:",
    ROTATION_GADGET_CYCLES,
)

print(
    "CNOT gadget cycles:",
    CNOT_GADGET_CYCLES,
)


GATE_RATE_ROWS = []


for p in P_VALUES:

    base = (
        base_history_channel(
            p
        )
    )


    rotation_channel = (
        channel_power(
            base,
            ROTATION_GADGET_CYCLES,
        )
    )


    cnot_channel = (
        channel_power(
            base,
            CNOT_GADGET_CYCLES,
        )
    )


    base_p = (
        1.0
        -
        base[
            0
        ]
    )


    rotation_p = (
        1.0
        -
        rotation_channel[
            0
        ]
    )


    cnot_p = (
        1.0
        -
        cnot_channel[
            0
        ]
    )


    GATE_RATE_ROWS.append(
        {
            "p":
                p,

            "base_history_pL":
                base_p,

            "rotation_gadget_pL":
                rotation_p,

            "cnot_participant_pL":
                cnot_p,
        }
    )


    print(
        f"p={p:.4f} | "
        f"base={base_p:.6f} | "
        f"rotation={rotation_p:.6f} | "
        f"CNOT-participant={cnot_p:.6f}"
    )


# ============================================================
# RESOURCE / OPERATION COUNT
# ============================================================

DATA_RY_COUNT = (
    N_UPLOADS
    *
    N_QUBITS
)

TRAINABLE_RY_COUNT = (
    N_UPLOADS
    *
    N_QUBITS
)

TRAINABLE_RZ_COUNT = (
    N_UPLOADS
    *
    N_QUBITS
)

LOGICAL_CNOT_COUNT = (
    N_UPLOADS
    *
    N_QUBITS
)


TOTAL_ROTATIONS = (
    DATA_RY_COUNT
    +
    TRAINABLE_RY_COUNT
    +
    TRAINABLE_RZ_COUNT
)


# Each logical CNOT has two logical-qubit participants.
CNOT_PARTICIPANT_COUNT = (
    2
    *
    LOGICAL_CNOT_COUNT
)


UNIFORM_CHANNEL_APPLICATIONS = (
    TOTAL_ROTATIONS
    +
    CNOT_PARTICIPANT_COUNT
)


GATE_AWARE_CHANNEL_APPLICATIONS = (
    TOTAL_ROTATIONS
    *
    ROTATION_GADGET_CYCLES
    +
    CNOT_PARTICIPANT_COUNT
    *
    CNOT_GADGET_CYCLES
)


N_LOGICAL_DATA_QUBITS = (
    N_QUBITS
)

N_PHYSICAL_DATA_QUBITS = (
    5
    *
    N_LOGICAL_DATA_QUBITS
)

# If all 8 blocks perform flagged EC simultaneously:
#
# 2 ancillas / block:
#     1 syndrome + 1 flag
#
# This is a simple parallel-block estimate, not a claim about
# the exact ancilla requirement of a joint pieceable CNOT.
PARALLEL_FLAG_ANCILLA_ESTIMATE = (
    2
    *
    N_LOGICAL_DATA_QUBITS
)

TOTAL_QUBIT_ESTIMATE = (
    N_PHYSICAL_DATA_QUBITS
    +
    PARALLEL_FLAG_ANCILLA_ESTIMATE
)


print("\n======================================")
print("LOGICAL CIRCUIT RESOURCE SUMMARY")
print("======================================")

print(
    "Data RY gates:",
    DATA_RY_COUNT,
)

print(
    "Trainable RY gates:",
    TRAINABLE_RY_COUNT,
)

print(
    "Trainable RZ gates:",
    TRAINABLE_RZ_COUNT,
)

print(
    "Logical CNOT gates:",
    LOGICAL_CNOT_COUNT,
)

print(
    "Total arbitrary rotations:",
    TOTAL_ROTATIONS,
)

print(
    "CNOT logical-qubit participants:",
    CNOT_PARTICIPANT_COUNT,
)

print(
    "Uniform channel applications:",
    UNIFORM_CHANNEL_APPLICATIONS,
)

print(
    "Gate-aware EC-equivalent channel applications:",
    GATE_AWARE_CHANNEL_APPLICATIONS,
)

print(
    "Logical qubits:",
    N_LOGICAL_DATA_QUBITS,
)

print(
    "[[5,1,3]] physical data qubits:",
    N_PHYSICAL_DATA_QUBITS,
)

print(
    "Parallel flag/syndrome ancilla estimate:",
    PARALLEL_FLAG_ANCILLA_ESTIMATE,
)

print(
    "Conceptual parallel qubit estimate:",
    TOTAL_QUBIT_ESTIMATE,
)


# ============================================================
# FAST 8-QUBIT LOGICAL STATEVECTOR
# ============================================================

def initial_state(
    batch_size,
):

    state = torch.zeros(
        batch_size,
        2 ** N_QUBITS,
        dtype=torch.complex64,
        device=device,
    )


    state[
        :,
        0,
    ] = (
        1.0
        +
        0.0j
    )


    return state


def apply_ry(
    state,
    theta,
    wire,
):

    batch_size = (
        state.shape[
            0
        ]
    )


    left = (
        2
        **
        wire
    )


    right = (
        2
        **
        (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[
        :,
        :,
        0,
        :,
    ]


    b = view[
        :,
        :,
        1,
        :,
    ]


    if not torch.is_tensor(
        theta
    ):

        theta = torch.tensor(
            theta,
            dtype=torch.float32,
            device=device,
        )


    theta = theta.to(
        dtype=torch.float32,
        device=device,
    )


    if theta.ndim == 0:

        theta = theta.expand(
            batch_size
        )


    theta = theta.reshape(
        batch_size,
        1,
        1,
    )


    c = torch.cos(
        theta
        /
        2
    )


    s = torch.sin(
        theta
        /
        2
    )


    return torch.stack(
        [
            c
            *
            a
            -
            s
            *
            b,

            s
            *
            a
            +
            c
            *
            b,
        ],
        dim=2,
    ).reshape(
        batch_size,
        -1,
    )


def apply_rz(
    state,
    theta,
    wire,
):

    batch_size = (
        state.shape[
            0
        ]
    )


    left = (
        2
        **
        wire
    )


    right = (
        2
        **
        (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[
        :,
        :,
        0,
        :,
    ]


    b = view[
        :,
        :,
        1,
        :,
    ]


    if not torch.is_tensor(
        theta
    ):

        theta = torch.tensor(
            theta,
            dtype=torch.float32,
            device=device,
        )


    theta = theta.to(
        dtype=torch.float32,
        device=device,
    )


    if theta.ndim == 0:

        theta = theta.expand(
            batch_size
        )


    theta = theta.reshape(
        batch_size,
        1,
        1,
    )


    phase0 = torch.exp(
        -0.5j
        *
        theta
    )


    phase1 = torch.exp(
        0.5j
        *
        theta
    )


    return torch.stack(
        [
            phase0
            *
            a,

            phase1
            *
            b,
        ],
        dim=2,
    ).reshape(
        batch_size,
        -1,
    )


def apply_x_fault(
    state,
    mask,
    wire,
):

    batch_size = (
        state.shape[
            0
        ]
    )


    left = (
        2
        **
        wire
    )


    right = (
        2
        **
        (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[
        :,
        :,
        0,
        :,
    ]


    b = view[
        :,
        :,
        1,
        :,
    ]


    mask = mask.bool().reshape(
        batch_size,
        1,
        1,
    )


    return torch.stack(
        [
            torch.where(
                mask,
                b,
                a,
            ),

            torch.where(
                mask,
                a,
                b,
            ),
        ],
        dim=2,
    ).reshape(
        batch_size,
        -1,
    )


def apply_z_fault(
    state,
    mask,
    wire,
):

    batch_size = (
        state.shape[
            0
        ]
    )


    left = (
        2
        **
        wire
    )


    right = (
        2
        **
        (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[
        :,
        :,
        0,
        :,
    ]


    b = view[
        :,
        :,
        1,
        :,
    ]


    mask = mask.bool().reshape(
        batch_size,
        1,
        1,
    )


    return torch.stack(
        [
            a,

            torch.where(
                mask,
                -b,
                b,
            ),
        ],
        dim=2,
    ).reshape(
        batch_size,
        -1,
    )


def apply_pauli_fault(
    state,
    pauli_codes,
    wire,
):

    x_mask = torch.logical_or(
        pauli_codes == 1,
        pauli_codes == 2,
    )


    z_mask = torch.logical_or(
        pauli_codes == 2,
        pauli_codes == 3,
    )


    state = apply_z_fault(
        state,
        z_mask,
        wire,
    )


    state = apply_x_fault(
        state,
        x_mask,
        wire,
    )


    return state


# ============================================================
# CNOT
# ============================================================

def build_cnot_permutation(
    control,
    target,
):

    indices = torch.arange(
        2 ** N_QUBITS,
        dtype=torch.long,
        device=device,
    )


    control_shift = (
        N_QUBITS
        -
        1
        -
        control
    )


    target_shift = (
        N_QUBITS
        -
        1
        -
        target
    )


    control_bit = (
        (
            indices
            >>
            control_shift
        )
        &
        1
    )


    return (
        indices
        ^
        (
            control_bit
            <<
            target_shift
        )
    )


CNOT_PERMUTATIONS = {}


for q in range(
    N_QUBITS
):

    control = q

    target = (
        q
        +
        1
    ) % N_QUBITS


    CNOT_PERMUTATIONS[
        (
            control,
            target,
        )
    ] = build_cnot_permutation(
        control,
        target,
    )


def apply_cnot(
    state,
    control,
    target,
):

    return state[
        :,
        CNOT_PERMUTATIONS[
            (
                control,
                target,
            )
        ],
    ]


# ============================================================
# Z EXPECTATION MATRIX
# ============================================================

basis_indices = torch.arange(
    2 ** N_QUBITS,
    dtype=torch.long,
    device=device,
)


z_signs = []


for q in range(
    N_QUBITS
):

    shift = (
        N_QUBITS
        -
        1
        -
        q
    )


    bit = (
        (
            basis_indices
            >>
            shift
        )
        &
        1
    )


    z_signs.append(
        1.0
        -
        2.0
        *
        bit.float()
    )


Z_SIGN_MATRIX = torch.stack(
    z_signs,
    dim=1,
)


# ============================================================
# SAMPLE PAULI CHANNEL
# ============================================================

def sample_pauli_codes(
    shape,
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )


    cumulative = np.cumsum(
        probabilities
    )


    random_values = torch.rand(
        shape,
        device=device,
    )


    codes = torch.zeros(
        shape,
        dtype=torch.uint8,
        device=device,
    )


    codes[
        random_values
        >=
        cumulative[
            0
        ]
    ] = 1


    codes[
        random_values
        >=
        cumulative[
            1
        ]
    ] = 2


    codes[
        random_values
        >=
        cumulative[
            2
        ]
    ] = 3


    return codes


# ============================================================
# LOGICAL CIRCUIT WITH GATE-SPECIFIC CHANNELS
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p,
    mode,
    rotation_cycles=None,
):

    batch_size = (
        inputs.shape[
            0
        ]
    )


    state = initial_state(
        batch_size
    )


    (
        data_rotation_channel,
        trainable_rotation_channel,
        cnot_channel,
    ) = gate_channels(
        p=p,
        mode=mode,
        rotation_cycles=rotation_cycles,
    )


    data_encoding_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            data_rotation_channel,
        )
    )


    trainable_ry_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            trainable_rotation_channel,
        )
    )


    trainable_rz_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            trainable_rotation_channel,
        )
    )


    cnot_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
                2,
            ),
            cnot_channel,
        )
    )


    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            *
            N_QUBITS
        )


        # ----------------------------------------------------
        # DATA-ENCODING RY
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            state = apply_ry(
                state,
                inputs[
                    :,
                    start
                    +
                    q,
                ],
                q,
            )


            state = apply_pauli_fault(
                state,
                data_encoding_errors[
                    :,
                    upload,
                    q,
                ],
                q,
            )


        # ----------------------------------------------------
        # TRAINABLE RY / RZ
        # ----------------------------------------------------

        for q in range(
            N_QUBITS
        ):

            state = apply_ry(
                state,
                quantum_weights[
                    upload,
                    q,
                    0,
                ],
                q,
            )


            state = apply_pauli_fault(
                state,
                trainable_ry_errors[
                    :,
                    upload,
                    q,
                ],
                q,
            )


            state = apply_rz(
                state,
                quantum_weights[
                    upload,
                    q,
                    1,
                ],
                q,
            )


            state = apply_pauli_fault(
                state,
                trainable_rz_errors[
                    :,
                    upload,
                    q,
                ],
                q,
            )


        # ----------------------------------------------------
        # LOGICAL CNOT RING
        #
        # The logical unitary remains CNOT.
        #
        # The gate-aware fault channel charges the
        # CNOT participant more EC-equivalent overhead than
        # the one-channel uniform model.
        # ----------------------------------------------------

        for gate_index in range(
            N_QUBITS
        ):

            control = (
                gate_index
            )


            target = (
                gate_index
                +
                1
            ) % N_QUBITS


            state = apply_cnot(
                state,
                control,
                target,
            )


            state = apply_pauli_fault(
                state,
                cnot_errors[
                    :,
                    upload,
                    gate_index,
                    0,
                ],
                control,
            )


            state = apply_pauli_fault(
                state,
                cnot_errors[
                    :,
                    upload,
                    gate_index,
                    1,
                ],
                target,
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
# EVALUATION
# ============================================================

def evaluate_classifier(
    p,
    mode,
    random_seed,
    rotation_cycles=None,
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
                non_blocking=True,
            )


            batch_size = (
                x.shape[
                    0
                ]
            )


            if (
                p == 0
                and
                mode == "unprotected"
            ):

                trajectories = 1

            else:

                trajectories = (
                    N_TRAJECTORIES
                )


            expanded_x = (
                x
                .unsqueeze(
                    1
                )
                .expand(
                    -1,
                    trajectories,
                    -1,
                )
                .reshape(
                    batch_size
                    *
                    trajectories,
                    32,
                )
            )


            quantum_features = (
                run_quantum_circuit(
                    inputs=expanded_x,
                    quantum_weights=(
                        model.quantum_weights
                    ),
                    p=p,
                    mode=mode,
                    rotation_cycles=rotation_cycles,
                )
            )


            if trajectories > 1:

                quantum_features = (
                    quantum_features
                    .reshape(
                        batch_size,
                        trajectories,
                        N_QUBITS,
                    )
                    .mean(
                        dim=1
                    )
                )


            logits = (
                model.classifier(
                    quantum_features
                )
            )


            predictions = torch.argmax(
                logits,
                dim=1,
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
        predictions_all,
    )


    macro_f1 = f1_score(
        labels_all,
        predictions_all,
        average="macro",
    )


    return (
        accuracy,
        macro_f1,
        elapsed,
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
    _,
) = evaluate_classifier(
    p=0.0,
    mode="unprotected",
    random_seed=SEED,
)


print(
    "Ideal accuracy:",
    f"{ideal_accuracy:.4f}",
)

print(
    "Ideal Macro-F1:",
    f"{ideal_f1:.4f}",
)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

MODES = [
    "unprotected",
    "uniform_history",
    "gate_aware_ft",
]


detailed_results = []


print("\n======================================")
print("GATE-AWARE FT CLASSIFIER EXPERIMENT")
print("======================================")

print(
    "Trajectories:",
    N_TRAJECTORIES,
)

print(
    "Seeds:",
    N_SEEDS,
)


for p_index, p in enumerate(
    P_VALUES
):

    print(
        "\n======================================"
    )

    print(
        f"Physical p = "
        f"{p:.4f}"
    )

    print(
        "======================================"
    )


    for seed_index in range(
        N_SEEDS
    ):

        base_seed = (
            SEED
            +
            10_000
            *
            p_index
            +
            seed_index
        )


        current_results = {}


        for mode_index, mode in enumerate(
            MODES
        ):

            if (
                mode == "unprotected"
                and
                p == 0
            ):

                accuracy = (
                    ideal_accuracy
                )

                macro_f1 = (
                    ideal_f1
                )

                elapsed = 0.0

            else:

                (
                    accuracy,
                    macro_f1,
                    elapsed,
                ) = evaluate_classifier(
                    p=p,
                    mode=mode,
                    random_seed=(
                        base_seed
                        +
                        100_000
                        *
                        mode_index
                    ),
                )


            current_results[
                mode
            ] = (
                accuracy,
                macro_f1,
                elapsed,
            )


        detailed_results.append(
            {
                "p":
                    p,

                "seed":
                    base_seed,

                "history_base_pL":
                    HISTORY_LOGICAL_ERROR[
                        p
                    ],

                "rotation_gadget_cycles":
                    ROTATION_GADGET_CYCLES,

                "cnot_gadget_cycles":
                    CNOT_GADGET_CYCLES,

                "unprotected_accuracy":
                    current_results[
                        "unprotected"
                    ][0],

                "uniform_history_accuracy":
                    current_results[
                        "uniform_history"
                    ][0],

                "gate_aware_ft_accuracy":
                    current_results[
                        "gate_aware_ft"
                    ][0],

                "unprotected_f1":
                    current_results[
                        "unprotected"
                    ][1],

                "uniform_history_f1":
                    current_results[
                        "uniform_history"
                    ][1],

                "gate_aware_ft_f1":
                    current_results[
                        "gate_aware_ft"
                    ][1],
            }
        )


        print(
            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "

            f"None="
            f"{current_results['unprotected'][0]:.4f} | "

            f"UniformHistory="
            f"{current_results['uniform_history'][0]:.4f} | "

            f"GateAwareFT="
            f"{current_results['gate_aware_ft'][0]:.4f}"
        )


# ============================================================
# ROTATION-OVERHEAD SENSITIVITY
#
# One classifier seed per rotation-cycle value to limit run
# time. This is a sensitivity analysis, not the main
# mean±std experiment.
# ============================================================

sensitivity_results = []


print("\n======================================")
print("ROTATION-OVERHEAD SENSITIVITY")
print("======================================")


for p_index, p in enumerate(
    P_VALUES
):

    for rotation_cycles in (
        ROTATION_CYCLE_SWEEP
    ):

        (
            accuracy,
            macro_f1,
            elapsed,
        ) = evaluate_classifier(
            p=p,
            mode="gate_aware_ft",
            random_seed=(
                SEED
                +
                8_000_000
                +
                1000
                *
                p_index
                +
                rotation_cycles
            ),
            rotation_cycles=rotation_cycles,
        )


        sensitivity_results.append(
            {
                "p":
                    p,

                "rotation_cycles":
                    rotation_cycles,

                "cnot_cycles":
                    CNOT_GADGET_CYCLES,

                "accuracy":
                    accuracy,

                "macro_f1":
                    macro_f1,

                "seconds":
                    elapsed,
            }
        )


        print(
            f"p={p:.4f} | "
            f"rotation cycles="
            f"{rotation_cycles} | "
            f"accuracy="
            f"{accuracy:.4f}"
        )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

DETAILED_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_ft_logical_gate_detailed.csv",
)


with open(
    DETAILED_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        detailed_results[
            0
        ].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        detailed_results
    )


SENSITIVITY_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_ft_rotation_overhead_sensitivity.csv",
)


with open(
    SENSITIVITY_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        sensitivity_results[
            0
        ].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        sensitivity_results
    )


GATE_RATE_CSV = os.path.join(
    RESULT_DIR,
    "ft_logical_gate_error_rates.csv",
)


with open(
    GATE_RATE_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        GATE_RATE_ROWS[
            0
        ].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        GATE_RATE_ROWS
    )


# ============================================================
# SUMMARY
# ============================================================

summary = []


for p in P_VALUES:

    subset = [
        row
        for row in detailed_results
        if row[
            "p"
        ] == p
    ]


    summary_row = {
        "p":
            p,

        "history_base_pL":
            HISTORY_LOGICAL_ERROR[
                p
            ],
    }


    for key in [
        "unprotected_accuracy",
        "uniform_history_accuracy",
        "gate_aware_ft_accuracy",
    ]:

        values = np.array(
            [
                row[
                    key
                ]
                for row in subset
            ]
        )


        summary_row[
            key
            +
            "_mean"
        ] = values.mean()


        summary_row[
            key
            +
            "_std"
        ] = (
            values.std(
                ddof=1
            )
            if len(
                values
            ) > 1
            else 0.0
        )


    summary.append(
        summary_row
    )


SUMMARY_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_ft_logical_gate_summary.csv",
)


with open(
    SUMMARY_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        summary[
            0
        ].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        summary
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Unprotected\t"
    "UniformHistory\t"
    "GateAwareFT"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['unprotected_accuracy_mean']:.4f}"
        f" ± "
        f"{row['unprotected_accuracy_std']:.4f}\t"

        f"{row['uniform_history_accuracy_mean']:.4f}"
        f" ± "
        f"{row['uniform_history_accuracy_std']:.4f}\t"

        f"{row['gate_aware_ft_accuracy_mean']:.4f}"
        f" ± "
        f"{row['gate_aware_ft_accuracy_std']:.4f}"
    )


# ============================================================
# PLOTS
# ============================================================

p_values = np.array(
    [
        row[
            "p"
        ]
        for row in summary
    ]
)


def summary_array(
    key,
):

    return np.array(
        [
            row[
                key
            ]
            for row in summary
        ]
    )


none_mean = summary_array(
    "unprotected_accuracy_mean"
)

none_std = summary_array(
    "unprotected_accuracy_std"
)


uniform_mean = summary_array(
    "uniform_history_accuracy_mean"
)

uniform_std = summary_array(
    "uniform_history_accuracy_std"
)


gate_aware_mean = summary_array(
    "gate_aware_ft_accuracy_mean"
)

gate_aware_std = summary_array(
    "gate_aware_ft_accuracy_std"
)


# ============================================================
# FIGURE 1
# ============================================================

plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.axhline(
    ideal_accuracy
    *
    100,
    linestyle="--",
    label=(
        f"Ideal classifier "
        f"({ideal_accuracy * 100:.2f}%)"
    ),
)


for mean, std, marker, label in [
    (
        none_mean,
        none_std,
        "o",
        "Unprotected",
    ),
    (
        uniform_mean,
        uniform_std,
        "s",
        "Uniform history channel",
    ),
    (
        gate_aware_mean,
        gate_aware_std,
        "^",
        "Gate-aware FT proxy",
    ),
]:

    plt.plot(
        p_values
        *
        100,
        mean
        *
        100,
        marker=marker,
        linewidth=2,
        label=label,
    )


    plt.fill_between(
        p_values
        *
        100,

        (
            mean
            -
            std
        )
        *
        100,

        (
            mean
            +
            std
        )
        *
        100,

        alpha=0.10,
    )


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Gate-Aware Fault-Tolerant Logical-Circuit Model"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_ft_logical_gate_accuracy.png",
)


plt.savefig(
    ACCURACY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2:
# BASE VS GATE-SPECIFIC LOGICAL ERROR
# ============================================================

base_rates = np.array(
    [
        row[
            "base_history_pL"
        ]
        for row in GATE_RATE_ROWS
    ]
)


rotation_rates = np.array(
    [
        row[
            "rotation_gadget_pL"
        ]
        for row in GATE_RATE_ROWS
    ]
)


cnot_rates = np.array(
    [
        row[
            "cnot_participant_pL"
        ]
        for row in GATE_RATE_ROWS
    ]
)


plt.figure(
    figsize=(
        8,
        5.8,
    )
)


plt.plot(
    p_values
    *
    100,
    base_rates
    *
    100,
    marker="o",
    linewidth=2,
    label="One history-decoded EC channel",
)


plt.plot(
    p_values
    *
    100,
    rotation_rates
    *
    100,
    marker="s",
    linewidth=2,
    label=(
        f"Rotation gadget "
        f"({ROTATION_GADGET_CYCLES} cycles)"
    ),
)


plt.plot(
    p_values
    *
    100,
    cnot_rates
    *
    100,
    marker="^",
    linewidth=2,
    label=(
        f"CNOT participant "
        f"({CNOT_GADGET_CYCLES} cycles)"
    ),
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Effective Logical Pauli Error Probability (%)"
)

plt.title(
    "Logical Error Cost of Fault-Tolerant Gate Gadgets"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GATE_ERROR_FIGURE = os.path.join(
    FIGURE_DIR,
    "ft_logical_gate_error_rates.png",
)


plt.savefig(
    GATE_ERROR_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3:
# ROTATION OVERHEAD SENSITIVITY
# ============================================================

plt.figure(
    figsize=(
        8.5,
        5.8,
    )
)


for rotation_cycles in (
    ROTATION_CYCLE_SWEEP
):

    values = []


    for p in P_VALUES:

        match = [
            row
            for row in sensitivity_results
            if (
                row[
                    "p"
                ] == p
                and
                row[
                    "rotation_cycles"
                ] == rotation_cycles
            )
        ]


        values.append(
            match[
                0
            ][
                "accuracy"
            ]
        )


    plt.plot(
        p_values
        *
        100,
        np.array(
            values
        )
        *
        100,
        marker="o",
        linewidth=2,
        label=(
            f"Rotation cycles = "
            f"{rotation_cycles}"
        ),
    )


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Sensitivity to Non-Clifford Rotation-Gadget Overhead"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


SENSITIVITY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_ft_rotation_overhead_sensitivity.png",
)


plt.savefig(
    SENSITIVITY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 4:
# GATE-AWARE PENALTY VS UNIFORM MODEL
# ============================================================

penalty = (
    gate_aware_mean
    -
    uniform_mean
)


plt.figure(
    figsize=(
        7.5,
        5.5,
    )
)


plt.plot(
    p_values
    *
    100,
    penalty
    *
    100,
    marker="o",
    linewidth=2,
)


plt.axhline(
    0,
    linestyle="--",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Gate-Aware − Uniform Accuracy "
    "(percentage points)"
)

plt.title(
    "Accuracy Cost of Logical-Gate Overhead"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


PENALTY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_ft_logical_gate_overhead_penalty.png",
)


plt.savefig(
    PENALTY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# SAVE RESOURCE REPORT
# ============================================================

RESOURCE_TXT = os.path.join(
    RESULT_DIR,
    "ft_logical_gate_resource_summary.txt",
)


with open(
    RESOURCE_TXT,
    "w",
) as file:

    file.write(
        "FAULT-TOLERANT LOGICAL-GATE RESOURCE SUMMARY\n"
    )

    file.write(
        "===========================================\n"
    )

    file.write(
        f"Logical qubits: "
        f"{N_LOGICAL_DATA_QUBITS}\n"
    )

    file.write(
        f"[[5,1,3]] physical data qubits: "
        f"{N_PHYSICAL_DATA_QUBITS}\n"
    )

    file.write(
        f"Parallel syndrome/flag ancilla estimate: "
        f"{PARALLEL_FLAG_ANCILLA_ESTIMATE}\n"
    )

    file.write(
        f"Conceptual parallel qubit estimate: "
        f"{TOTAL_QUBIT_ESTIMATE}\n\n"
    )

    file.write(
        f"Data RY operations: "
        f"{DATA_RY_COUNT}\n"
    )

    file.write(
        f"Trainable RY operations: "
        f"{TRAINABLE_RY_COUNT}\n"
    )

    file.write(
        f"Trainable RZ operations: "
        f"{TRAINABLE_RZ_COUNT}\n"
    )

    file.write(
        f"Logical CNOT operations: "
        f"{LOGICAL_CNOT_COUNT}\n\n"
    )

    file.write(
        f"Rotation gadget EC-equivalent cycles: "
        f"{ROTATION_GADGET_CYCLES}\n"
    )

    file.write(
        f"CNOT gadget EC-equivalent cycles: "
        f"{CNOT_GADGET_CYCLES}\n"
    )

    file.write(
        f"Uniform block-channel applications: "
        f"{UNIFORM_CHANNEL_APPLICATIONS}\n"
    )

    file.write(
        f"Gate-aware block-channel applications: "
        f"{GATE_AWARE_CHANNEL_APPLICATIONS}\n\n"
    )

    file.write(
        "IMPORTANT: these are modeling/resource proxies, "
        "not exact compiled hardware gate counts.\n"
    )


# ============================================================
# DONE
# ============================================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Detailed CSV:",
    DETAILED_CSV,
)

print(
    "Summary CSV:",
    SUMMARY_CSV,
)

print(
    "Gate-rate CSV:",
    GATE_RATE_CSV,
)

print(
    "Sensitivity CSV:",
    SENSITIVITY_CSV,
)

print(
    "Resource summary:",
    RESOURCE_TXT,
)

print(
    "Accuracy figure:",
    ACCURACY_FIGURE,
)

print(
    "Gate-error figure:",
    GATE_ERROR_FIGURE,
)

print(
    "Rotation sensitivity figure:",
    SENSITIVITY_FIGURE,
)

print(
    "Overhead penalty figure:",
    PENALTY_FIGURE,
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "UniformHistory applies one syndrome-history "
    "logical Pauli channel at every modeled logical "
    "fault location."
)

print(
    "GateAwareFT charges arbitrary RY/RZ rotations "
    "and logical CNOT participants additional "
    "EC-equivalent channel overhead."
)

print(
    "The [[5,1,3]] code supports fault-tolerant "
    "single-qubit Clifford operations, but arbitrary "
    "RY/RZ rotations are non-Clifford and are not "
    "transversal."
)

print(
    "The entangling-gate model is a pieceable-FT "
    "overhead proxy rather than an explicit simulation "
    "of the full joint [[10,2,3]] intermediate decoder."
)

print(
    "The syndrome-history CSV stores only total pL, so "
    "this stage symmetrizes the residual logical channel "
    "as X=Y=Z=pL/3."
)

print(
    "The next stage should assemble the final end-to-end "
    "FT classifier/resource model and freeze the simulator "
    "configuration before the LaTeX report and Braket/Rigetti "
    "hardware preparation."
)
