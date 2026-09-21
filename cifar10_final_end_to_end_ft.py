import os
import csv
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
)


# ============================================================
# FINAL END-TO-END FT CIFAR-10 SIMULATOR
#
# This is the final simulator-stage script before preparing the
# LaTeX report and then moving to the Amazon Braket / Rigetti
# hardware-preparation stage.
#
# It freezes the full modeling pipeline:
#
# CIFAR-10
#   -> ResNet18 learned 32-D features
#   -> 8 logical-qubit re-uploading VQC
#   -> [[5,1,3]] logical protection
#   -> best fixed syndrome decoder selected from the previous
#      decoder study
#   -> gate-aware FT overhead for arbitrary rotations and
#      logical entangling gates
#   -> 10-class classical output head
#
# It reports:
#
#   * ideal classifier performance
#   * unprotected noisy performance
#   * uniform protected logical-channel performance
#   * final gate-aware FT performance
#   * logical-gadget error rates
#   * expected logical fault-event count
#   * 77% target-accuracy operating region
#   * approximate FT/unprotected break-even point
#   * per-class recalls at a report operating point
#   * confusion matrices
#   * resource summary
#
# IMPORTANT SCIENTIFIC SCOPE
# --------------------------
# The code is still a LOGICAL / RESOURCE-AWARE FT MODEL.
#
# It does not explicitly simulate a 40+ physical-qubit
# statevector.
#
# The residual logical channel from the decoder stage is
# represented by its total p_L and symmetrized as
#
#     P(X_L) = P(Y_L) = P(Z_L) = p_L / 3.
#
# The logical rotation and CNOT gadgets are represented by
# repeated applications of the protected logical channel.
# This quantifies FT overhead, but it is NOT an exact compiled
# physical Clifford+T / injection / pieceable-CNOT circuit.
#
# That hardware-specific compilation is intentionally left for
# the Braket/Rigetti stage.
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

TARGET_ACCURACY = 0.77

# Report per-class metrics at this physical noise value.
REPORT_P = 0.010


# ============================================================
# FINAL FROZEN FT-GADGET ASSUMPTIONS
# ============================================================
#
# These are the same modest gate-overhead assumptions used in
# the preceding logical-gate study.
#
# One "cycle" = one application of the selected protected
# logical Pauli channel.
# ============================================================

ROTATION_GADGET_CYCLES = 2
CNOT_GADGET_CYCLES = 2


# ============================================================
# DECODER SELECTION
# ============================================================
#
# "auto_global":
#     choose ONE decoder for the entire experiment by the
#     lowest mean logical error over the full p sweep.
#
# This avoids choosing a different decoder at every p.
#
# Other choices:
#     "previous_flagged"
#     "majority3"
#     "history3"
# ============================================================

DECODER_SELECTION = "auto_global"


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
# LOAD CIFAR-10 FEATURES
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True,
)

X_test = data[
    "X_test"
].astype(np.float32)

y_test = data[
    "y_test"
].astype(np.int64)


test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


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
    "truck",
]


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

    def __init__(self):

        super().__init__()

        self.quantum_weights = nn.Parameter(
            0.05
            * torch.randn(
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
            nn.Dropout(0.10),
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


model = model.to(device)


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
# LOAD PREVIOUS DECODER STUDY
# ============================================================

if not os.path.exists(
    HISTORY_CHANNEL_CSV
):

    raise FileNotFoundError(
        "\nRequired previous result not found:\n"
        f"{HISTORY_CHANNEL_CSV}\n\n"
        "Run cifar10_5qubit_syndrome_history.py first."
    )


decoder_rows = []


with open(
    HISTORY_CHANNEL_CSV,
    "r",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        decoder_rows.append(
            {
                "p":
                    float(
                        row["p"]
                    ),

                "ideal":
                    float(
                        row[
                            "ideal_logical_error"
                        ]
                    ),

                "previous_flagged":
                    float(
                        row[
                            "previous_flagged_logical_error"
                        ]
                    ),

                "majority3":
                    float(
                        row[
                            "majority3_logical_error"
                        ]
                    ),

                "history3":
                    float(
                        row[
                            "history3_logical_error"
                        ]
                    ),

                "history_coset_accuracy":
                    float(
                        row[
                            "history_coset_prediction_accuracy"
                        ]
                    ),

                "flag_rate":
                    float(
                        row[
                            "three_round_flag_cycle_rate"
                        ]
                    ),

                "temporal_change_rate":
                    float(
                        row[
                            "temporal_syndrome_change_rate"
                        ]
                    ),
            }
        )


decoder_rows = sorted(
    decoder_rows,
    key=lambda x: x["p"],
)


P_VALUES = [
    row["p"]
    for row in decoder_rows
]


# ============================================================
# SELECT ONE FIXED FINAL DECODER
# ============================================================

candidate_decoders = [
    "previous_flagged",
    "majority3",
    "history3",
]


if DECODER_SELECTION == "auto_global":

    decoder_mean_pl = {}

    for decoder_name in candidate_decoders:

        decoder_mean_pl[
            decoder_name
        ] = float(
            np.mean(
                [
                    row[
                        decoder_name
                    ]
                    for row in decoder_rows
                ]
            )
        )


    FINAL_DECODER = min(
        decoder_mean_pl,
        key=decoder_mean_pl.get,
    )

else:

    if DECODER_SELECTION not in candidate_decoders:

        raise ValueError(
            "DECODER_SELECTION must be "
            "'auto_global', 'previous_flagged', "
            "'majority3', or 'history3'."
        )

    FINAL_DECODER = (
        DECODER_SELECTION
    )


FINAL_LOGICAL_ERROR = {
    row["p"]:
        row[
            FINAL_DECODER
        ]
    for row in decoder_rows
}


print("\n======================================")
print("FINAL DECODER SELECTION")
print("======================================")

print(
    "Selection mode:",
    DECODER_SELECTION,
)

if DECODER_SELECTION == "auto_global":

    for decoder_name in candidate_decoders:

        print(
            f"{decoder_name}: "
            f"mean pL="
            f"{decoder_mean_pl[decoder_name]:.6f}"
        )


print(
    "Selected fixed decoder:",
    FINAL_DECODER,
)


print("\nSelected decoder pL values:")

for p in P_VALUES:

    print(
        f"p={p:.4f} | "
        f"pL={FINAL_LOGICAL_ERROR[p]:.6f}"
    )


# ============================================================
# PAULI CHANNEL UTILITIES
# ============================================================

# I, X, Y, Z represented by symplectic (x,z) bits.
PAULI_BITS = [
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
]


BITS_TO_PAULI = {
    bits: index
    for index, bits in enumerate(
        PAULI_BITS
    )
}


def symmetric_logical_channel(
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
            1.0 - p_logical,
            p_logical / 3.0,
            p_logical / 3.0,
            p_logical / 3.0,
        ],
        dtype=np.float64,
    )


def physical_depolarizing_channel(
    p,
):

    return np.array(
        [
            1.0 - p,
            p / 3.0,
            p / 3.0,
            p / 3.0,
        ],
        dtype=np.float64,
    )


def compose_pauli_channels(
    first,
    second,
):

    output = np.zeros(
        4,
        dtype=np.float64,
    )


    for i in range(4):

        xi, zi = PAULI_BITS[i]

        for j in range(4):

            xj, zj = PAULI_BITS[j]

            k = BITS_TO_PAULI[
                (
                    xi ^ xj,
                    zi ^ zj,
                )
            ]

            output[k] += (
                first[i]
                * second[j]
            )


    return (
        output
        / output.sum()
    )


def channel_power(
    channel,
    cycles,
):

    output = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )


    for _ in range(cycles):

        output = compose_pauli_channels(
            output,
            channel,
        )


    return output


# ============================================================
# FINAL CHANNELS
# ============================================================

def base_protected_channel(
    p,
):

    return symmetric_logical_channel(
        FINAL_LOGICAL_ERROR[p]
    )


def channels_for_mode(
    p,
    mode,
):

    if mode == "unprotected":

        physical = (
            physical_depolarizing_channel(
                p
            )
        )

        return (
            physical,
            physical,
            physical,
        )


    base = (
        base_protected_channel(
            p
        )
    )


    if mode == "uniform_protected":

        return (
            base,
            base,
            base,
        )


    if mode == "final_ft":

        rotation_channel = (
            channel_power(
                base,
                ROTATION_GADGET_CYCLES,
            )
        )


        cnot_participant_channel = (
            channel_power(
                base,
                CNOT_GADGET_CYCLES,
            )
        )


        return (
            rotation_channel,
            rotation_channel,
            cnot_participant_channel,
        )


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# CIRCUIT / RESOURCE COUNTS
# ============================================================

DATA_RY_COUNT = (
    N_UPLOADS
    * N_QUBITS
)

TRAINABLE_RY_COUNT = (
    N_UPLOADS
    * N_QUBITS
)

TRAINABLE_RZ_COUNT = (
    N_UPLOADS
    * N_QUBITS
)

LOGICAL_CNOT_COUNT = (
    N_UPLOADS
    * N_QUBITS
)


TOTAL_ROTATIONS = (
    DATA_RY_COUNT
    +
    TRAINABLE_RY_COUNT
    +
    TRAINABLE_RZ_COUNT
)


CNOT_PARTICIPANT_COUNT = (
    2
    * LOGICAL_CNOT_COUNT
)


UNIFORM_LOGICAL_LOCATIONS = (
    TOTAL_ROTATIONS
    +
    CNOT_PARTICIPANT_COUNT
)


FT_EQUIVALENT_CYCLES = (
    TOTAL_ROTATIONS
    * ROTATION_GADGET_CYCLES
    +
    CNOT_PARTICIPANT_COUNT
    * CNOT_GADGET_CYCLES
)


LOGICAL_QUBITS = N_QUBITS

PHYSICAL_DATA_QUBITS = (
    5
    * LOGICAL_QUBITS
)

# One syndrome + one flag ancilla per simultaneously corrected
# logical block.
BLOCKWISE_ANCILLA_ESTIMATE = (
    2
    * LOGICAL_QUBITS
)

CONCEPTUAL_PARALLEL_QUBITS = (
    PHYSICAL_DATA_QUBITS
    +
    BLOCKWISE_ANCILLA_ESTIMATE
)


print("\n======================================")
print("FINAL RESOURCE MODEL")
print("======================================")

print(
    "Logical qubits:",
    LOGICAL_QUBITS,
)

print(
    "Physical data qubits:",
    PHYSICAL_DATA_QUBITS,
)

print(
    "Blockwise syndrome/flag ancillas:",
    BLOCKWISE_ANCILLA_ESTIMATE,
)

print(
    "Conceptual parallel qubits:",
    CONCEPTUAL_PARALLEL_QUBITS,
)

print(
    "Data RY:",
    DATA_RY_COUNT,
)

print(
    "Trainable RY:",
    TRAINABLE_RY_COUNT,
)

print(
    "Trainable RZ:",
    TRAINABLE_RZ_COUNT,
)

print(
    "Logical CNOT:",
    LOGICAL_CNOT_COUNT,
)

print(
    "Uniform logical locations:",
    UNIFORM_LOGICAL_LOCATIONS,
)

print(
    "FT-equivalent protected cycles:",
    FT_EQUIVALENT_CYCLES,
)


# ============================================================
# EFFECTIVE LOGICAL-GADGET ERROR TABLE
# ============================================================

gadget_rows = []


for p in P_VALUES:

    base = base_protected_channel(
        p
    )

    rotation = channel_power(
        base,
        ROTATION_GADGET_CYCLES,
    )

    cnot = channel_power(
        base,
        CNOT_GADGET_CYCLES,
    )


    base_pl = (
        1.0 - base[0]
    )

    rotation_pl = (
        1.0 - rotation[0]
    )

    cnot_pl = (
        1.0 - cnot[0]
    )


    expected_ft_events = (
        TOTAL_ROTATIONS
        * rotation_pl
        +
        CNOT_PARTICIPANT_COUNT
        * cnot_pl
    )


    gadget_rows.append(
        {
            "p":
                p,

            "base_protected_pL":
                base_pl,

            "rotation_gadget_pL":
                rotation_pl,

            "cnot_participant_pL":
                cnot_pl,

            "expected_nonidentity_logical_events":
                expected_ft_events,
        }
    )


print("\n======================================")
print("FINAL LOGICAL-GADGET ERROR MODEL")
print("======================================")

for row in gadget_rows:

    print(
        f"p={row['p']:.4f} | "
        f"base={row['base_protected_pL']:.6f} | "
        f"rotation={row['rotation_gadget_pL']:.6f} | "
        f"CNOT-participant={row['cnot_participant_pL']:.6f} | "
        f"E[fault events]="
        f"{row['expected_nonidentity_logical_events']:.3f}"
    )


# ============================================================
# FAST LOGICAL STATEVECTOR
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

    state[:, 0] = (
        1.0 + 0.0j
    )

    return state


def apply_ry(
    state,
    theta,
    wire,
):

    batch_size = (
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2
        **
        (
            N_QUBITS
            - wire
            - 1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[:, :, 0, :]
    b = view[:, :, 1, :]


    if not torch.is_tensor(theta):

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
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )


    return torch.stack(
        [
            c * a - s * b,
            s * a + c * b,
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
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2
        **
        (
            N_QUBITS
            - wire
            - 1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[:, :, 0, :]
    b = view[:, :, 1, :]


    if not torch.is_tensor(theta):

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
        -0.5j * theta
    )

    phase1 = torch.exp(
        0.5j * theta
    )


    return torch.stack(
        [
            phase0 * a,
            phase1 * b,
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
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2
        **
        (
            N_QUBITS
            - wire
            - 1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[:, :, 0, :]
    b = view[:, :, 1, :]


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
        state.shape[0]
    )

    left = (
        2 ** wire
    )

    right = (
        2
        **
        (
            N_QUBITS
            - wire
            - 1
        )
    )


    view = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = view[:, :, 0, :]
    b = view[:, :, 1, :]


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
        - 1
        - control
    )

    target_shift = (
        N_QUBITS
        - 1
        - target
    )


    control_bit = (
        (
            indices
            >> control_shift
        )
        & 1
    )


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
        - 1
        - q
    )


    bit = (
        (
            basis_indices
            >> shift
        )
        & 1
    )


    z_signs.append(
        1.0
        - 2.0
        * bit.float()
    )


Z_SIGN_MATRIX = torch.stack(
    z_signs,
    dim=1,
)


# ============================================================
# SAMPLE A PAULI CHANNEL
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


    r = torch.rand(
        shape,
        device=device,
    )


    codes = torch.zeros(
        shape,
        dtype=torch.uint8,
        device=device,
    )


    codes[
        r >= cumulative[0]
    ] = 1


    codes[
        r >= cumulative[1]
    ] = 2


    codes[
        r >= cumulative[2]
    ] = 3


    return codes


# ============================================================
# LOGICAL CIRCUIT
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p,
    mode,
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    (
        data_rotation_channel,
        trainable_rotation_channel,
        cnot_channel,
    ) = channels_for_mode(
        p,
        mode,
    )


    encoding_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            data_rotation_channel,
        )
    )


    ry_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            trainable_rotation_channel,
        )
    )


    rz_errors = (
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
            * N_QUBITS
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
                    start + q,
                ],
                q,
            )


            state = apply_pauli_fault(
                state,
                encoding_errors[
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
                ry_errors[
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
                rz_errors[
                    :,
                    upload,
                    q,
                ],
                q,
            )


        # ----------------------------------------------------
        # LOGICAL CNOT RING
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
        @ Z_SIGN_MATRIX
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_classifier(
    p,
    mode,
    random_seed,
    return_predictions=False,
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
                x.shape[0]
            )


            # Only the unprotected p=0 circuit is exactly
            # noiseless in this final model.
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
                .unsqueeze(1)
                .expand(
                    -1,
                    trajectories,
                    -1,
                )
                .reshape(
                    batch_size
                    * trajectories,
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


            logits = model.classifier(
                quantum_features
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
        - start_time
    )


    predictions_all = np.asarray(
        predictions_all,
        dtype=np.int64,
    )

    labels_all = np.asarray(
        labels_all,
        dtype=np.int64,
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


    if return_predictions:

        return (
            accuracy,
            macro_f1,
            elapsed,
            labels_all,
            predictions_all,
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
print("IDEAL CLASSIFIER BASELINE")
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
    f"Ideal accuracy = "
    f"{ideal_accuracy:.4f}"
)

print(
    f"Ideal Macro-F1 = "
    f"{ideal_f1:.4f}"
)


# ============================================================
# MAIN FINAL EXPERIMENT
# ============================================================

MODES = [
    "unprotected",
    "uniform_protected",
    "final_ft",
]


detailed_results = []


print("\n======================================")
print("FINAL END-TO-END FT EXPERIMENT")
print("======================================")

print(
    "Selected decoder:",
    FINAL_DECODER,
)

print(
    "Trajectories:",
    N_TRAJECTORIES,
)

print(
    "Seeds:",
    N_SEEDS,
)

print(
    "Target accuracy:",
    TARGET_ACCURACY,
)


for p_index, p in enumerate(
    P_VALUES
):

    print(
        "\n======================================"
    )

    print(
        f"Physical depolarizing p = "
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
            * p_index
            +
            seed_index
        )


        mode_results = {}


        for mode_index, mode in enumerate(
            MODES
        ):

            if (
                p == 0
                and
                mode == "unprotected"
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
                        * mode_index
                    ),
                )


            mode_results[
                mode
            ] = {
                "accuracy":
                    accuracy,

                "f1":
                    macro_f1,

                "seconds":
                    elapsed,
            }


        detailed_results.append(
            {
                "p":
                    p,

                "seed":
                    base_seed,

                "selected_decoder":
                    FINAL_DECODER,

                "base_logical_error":
                    FINAL_LOGICAL_ERROR[
                        p
                    ],

                "rotation_gadget_cycles":
                    ROTATION_GADGET_CYCLES,

                "cnot_gadget_cycles":
                    CNOT_GADGET_CYCLES,

                "unprotected_accuracy":
                    mode_results[
                        "unprotected"
                    ][
                        "accuracy"
                    ],

                "uniform_accuracy":
                    mode_results[
                        "uniform_protected"
                    ][
                        "accuracy"
                    ],

                "final_ft_accuracy":
                    mode_results[
                        "final_ft"
                    ][
                        "accuracy"
                    ],

                "unprotected_f1":
                    mode_results[
                        "unprotected"
                    ][
                        "f1"
                    ],

                "uniform_f1":
                    mode_results[
                        "uniform_protected"
                    ][
                        "f1"
                    ],

                "final_ft_f1":
                    mode_results[
                        "final_ft"
                    ][
                        "f1"
                    ],
            }
        )


        print(
            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "

            f"None="
            f"{mode_results['unprotected']['accuracy']:.4f} | "

            f"Uniform="
            f"{mode_results['uniform_protected']['accuracy']:.4f} | "

            f"FinalFT="
            f"{mode_results['final_ft']['accuracy']:.4f}"
        )


# ============================================================
# SAVE DETAILED CSV
# ============================================================

DETAILED_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_final_ft_detailed.csv",
)


with open(
    DETAILED_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        detailed_results[0].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        detailed_results
    )


# ============================================================
# SUMMARY
# ============================================================

summary = []


for p in P_VALUES:

    subset = [
        row
        for row in detailed_results
        if row["p"] == p
    ]


    summary_row = {
        "p":
            p,

        "base_logical_error":
            FINAL_LOGICAL_ERROR[
                p
            ],
    }


    for key in [
        "unprotected_accuracy",
        "uniform_accuracy",
        "final_ft_accuracy",
        "unprotected_f1",
        "uniform_f1",
        "final_ft_f1",
    ]:

        values = np.array(
            [
                row[key]
                for row in subset
            ]
        )


        summary_row[
            key + "_mean"
        ] = values.mean()


        summary_row[
            key + "_std"
        ] = (
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )


    summary_row[
        "final_ft_gain_vs_unprotected"
    ] = (
        summary_row[
            "final_ft_accuracy_mean"
        ]
        -
        summary_row[
            "unprotected_accuracy_mean"
        ]
    )


    summary_row[
        "final_ft_penalty_vs_uniform"
    ] = (
        summary_row[
            "final_ft_accuracy_mean"
        ]
        -
        summary_row[
            "uniform_accuracy_mean"
        ]
    )


    summary_row[
        "final_ft_accuracy_retention"
    ] = (
        summary_row[
            "final_ft_accuracy_mean"
        ]
        /
        ideal_accuracy
    )


    summary_row[
        "unprotected_meets_77"
    ] = int(
        summary_row[
            "unprotected_accuracy_mean"
        ]
        >= TARGET_ACCURACY
    )


    summary_row[
        "final_ft_meets_77"
    ] = int(
        summary_row[
            "final_ft_accuracy_mean"
        ]
        >= TARGET_ACCURACY
    )


    summary.append(
        summary_row
    )


SUMMARY_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_final_ft_summary.csv",
)


with open(
    SUMMARY_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        summary[0].keys()
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
# OPERATING-POINT ANALYSIS
# ============================================================

def max_p_meeting_target(
    mode_key,
):

    valid = [
        row["p"]
        for row in summary
        if row[
            mode_key
        ] >= TARGET_ACCURACY
    ]


    if not valid:

        return None


    return max(
        valid
    )


def approximate_break_even():

    differences = np.array(
        [
            row[
                "final_ft_accuracy_mean"
            ]
            -
            row[
                "unprotected_accuracy_mean"
            ]
            for row in summary
        ]
    )


    p_array = np.array(
        [
            row["p"]
            for row in summary
        ]
    )


    for i in range(
        len(p_array) - 1
    ):

        d0 = differences[i]
        d1 = differences[
            i + 1
        ]


        if d0 == 0:

            return float(
                p_array[i]
            )


        if (
            d0
            * d1
            < 0
        ):

            p0 = p_array[i]
            p1 = p_array[
                i + 1
            ]


            alpha = (
                -d0
                /
                (
                    d1
                    - d0
                )
            )


            return float(
                p0
                +
                alpha
                * (
                    p1
                    - p0
                )
            )


    return None


max_unprotected_77 = (
    max_p_meeting_target(
        "unprotected_accuracy_mean"
    )
)

max_final_ft_77 = (
    max_p_meeting_target(
        "final_ft_accuracy_mean"
    )
)

break_even_p = (
    approximate_break_even()
)


print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Unprotected\t"
    "UniformProtected\t"
    "FinalFT"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['unprotected_accuracy_mean']:.4f}"
        f" ± "
        f"{row['unprotected_accuracy_std']:.4f}\t"

        f"{row['uniform_accuracy_mean']:.4f}"
        f" ± "
        f"{row['uniform_accuracy_std']:.4f}\t"

        f"{row['final_ft_accuracy_mean']:.4f}"
        f" ± "
        f"{row['final_ft_accuracy_std']:.4f}"
    )


print("\n======================================")
print("TARGET / BREAK-EVEN SUMMARY")
print("======================================")

print(
    "Target accuracy:",
    TARGET_ACCURACY,
)


print(
    "Largest tested p with "
    "unprotected accuracy >= target:",
    max_unprotected_77,
)


print(
    "Largest tested p with "
    "FinalFT accuracy >= target:",
    max_final_ft_77,
)


print(
    "Approximate FinalFT/unprotected "
    "break-even p:",
    break_even_p,
)


# ============================================================
# REPORT-POINT PER-CLASS ANALYSIS
# ============================================================

if REPORT_P not in P_VALUES:

    report_p = min(
        P_VALUES,
        key=lambda x: abs(
            x - REPORT_P
        ),
    )

else:

    report_p = REPORT_P


print("\n======================================")
print("PER-CLASS REPORT POINT")
print("======================================")

print(
    "Physical p:",
    report_p,
)


report_modes = [
    "unprotected",
    "final_ft",
]


per_class_rows = []


for mode_index, mode in enumerate(
    report_modes
):

    (
        report_accuracy,
        report_f1,
        _,
        labels,
        predictions,
    ) = evaluate_classifier(
        p=report_p,
        mode=mode,
        random_seed=(
            SEED
            +
            9_000_000
            +
            mode_index
        ),
        return_predictions=True,
    )


    recalls = recall_score(
        labels,
        predictions,
        labels=np.arange(
            N_CLASSES
        ),
        average=None,
        zero_division=0,
    )


    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(
            N_CLASSES
        ),
    )


    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        per_class_rows.append(
            {
                "p":
                    report_p,

                "mode":
                    mode,

                "class_index":
                    class_index,

                "class_name":
                    class_name,

                "recall":
                    recalls[
                        class_index
                    ],

                "overall_accuracy":
                    report_accuracy,

                "macro_f1":
                    report_f1,
            }
        )


    plt.figure(
        figsize=(
            8,
            7,
        )
    )


    plt.imshow(
        cm,
        aspect="auto",
    )


    plt.colorbar(
        label="Count"
    )


    plt.xticks(
        np.arange(
            N_CLASSES
        ),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )


    plt.yticks(
        np.arange(
            N_CLASSES
        ),
        CLASS_NAMES,
    )


    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "True class"
    )


    plt.title(
        (
            f"{mode} confusion matrix "
            f"(p={report_p:.3f})"
        )
    )


    plt.tight_layout()


    p_name = str(
        report_p
    ).replace(
        ".",
        "p",
    )


    confusion_path = os.path.join(
        FIGURE_DIR,
        (
            f"cifar10_final_{mode}_"
            f"confusion_p_{p_name}.png"
        ),
    )


    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


PER_CLASS_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_final_ft_per_class.csv",
)


with open(
    PER_CLASS_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        per_class_rows[0].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        per_class_rows
    )


# ============================================================
# FINAL FIGURE 1:
# ACCURACY
# ============================================================

p_array = np.array(
    [
        row["p"]
        for row in summary
    ]
)


none_mean = np.array(
    [
        row[
            "unprotected_accuracy_mean"
        ]
        for row in summary
    ]
)

none_std = np.array(
    [
        row[
            "unprotected_accuracy_std"
        ]
        for row in summary
    ]
)


uniform_mean = np.array(
    [
        row[
            "uniform_accuracy_mean"
        ]
        for row in summary
    ]
)

uniform_std = np.array(
    [
        row[
            "uniform_accuracy_std"
        ]
        for row in summary
    ]
)


ft_mean = np.array(
    [
        row[
            "final_ft_accuracy_mean"
        ]
        for row in summary
    ]
)

ft_std = np.array(
    [
        row[
            "final_ft_accuracy_std"
        ]
        for row in summary
    ]
)


plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.axhline(
    ideal_accuracy
    * 100,
    linestyle="--",
    label=(
        f"Ideal "
        f"({ideal_accuracy * 100:.2f}%)"
    ),
)


plt.axhline(
    TARGET_ACCURACY
    * 100,
    linestyle=":",
    label=(
        f"Target "
        f"({TARGET_ACCURACY * 100:.0f}%)"
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
        "Uniform protected",
    ),
    (
        ft_mean,
        ft_std,
        "^",
        "Final gate-aware FT",
    ),
]:

    plt.plot(
        p_array
        * 100,
        mean
        * 100,
        marker=marker,
        linewidth=2,
        label=label,
    )


    plt.fill_between(
        p_array
        * 100,

        (
            mean
            - std
        )
        * 100,

        (
            mean
            + std
        )
        * 100,

        alpha=0.10,
    )


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Final End-to-End Fault-Tolerant CIFAR-10 Model"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


FINAL_ACCURACY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_final_ft_accuracy.png",
)


plt.savefig(
    FINAL_ACCURACY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL FIGURE 2:
# ACCURACY GAIN VS UNPROTECTED
# ============================================================

gain = (
    ft_mean
    - none_mean
)


plt.figure(
    figsize=(
        7.8,
        5.5,
    )
)


plt.plot(
    p_array
    * 100,
    gain
    * 100,
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
    "FinalFT − Unprotected Accuracy "
    "(percentage points)"
)

plt.title(
    "End-Task Benefit of Fault-Tolerant Protection"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GAIN_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_final_ft_gain_vs_unprotected.png",
)


plt.savefig(
    GAIN_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL FIGURE 3:
# FT OVERHEAD PENALTY
# ============================================================

overhead_penalty = (
    ft_mean
    - uniform_mean
)


plt.figure(
    figsize=(
        7.8,
        5.5,
    )
)


plt.plot(
    p_array
    * 100,
    overhead_penalty
    * 100,
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
    "FinalFT − Uniform Protected Accuracy "
    "(percentage points)"
)

plt.title(
    "Accuracy Cost of Fault-Tolerant Gate Overhead"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


OVERHEAD_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_final_ft_overhead_penalty.png",
)


plt.savefig(
    OVERHEAD_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL FIGURE 4:
# LOGICAL-GADGET ERROR RATES
# ============================================================

base_pl = np.array(
    [
        row[
            "base_protected_pL"
        ]
        for row in gadget_rows
    ]
)


rotation_pl = np.array(
    [
        row[
            "rotation_gadget_pL"
        ]
        for row in gadget_rows
    ]
)


cnot_pl = np.array(
    [
        row[
            "cnot_participant_pL"
        ]
        for row in gadget_rows
    ]
)


plt.figure(
    figsize=(
        8,
        5.8,
    )
)


plt.plot(
    p_array
    * 100,
    base_pl
    * 100,
    marker="o",
    linewidth=2,
    label="Selected protected EC channel",
)


plt.plot(
    p_array
    * 100,
    rotation_pl
    * 100,
    marker="s",
    linewidth=2,
    label="Logical rotation gadget",
)


plt.plot(
    p_array
    * 100,
    cnot_pl
    * 100,
    marker="^",
    linewidth=2,
    label="Logical CNOT participant",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Effective Logical Pauli Error Probability (%)"
)

plt.title(
    "Final Gate-Aware Logical Error Model"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GADGET_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_final_ft_gadget_error_rates.png",
)


plt.savefig(
    GADGET_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL FIGURE 5:
# EXPECTED NON-IDENTITY LOGICAL EVENTS
# ============================================================

expected_events = np.array(
    [
        row[
            "expected_nonidentity_logical_events"
        ]
        for row in gadget_rows
    ]
)


plt.figure(
    figsize=(
        7.8,
        5.5,
    )
)


plt.plot(
    p_array
    * 100,
    expected_events,
    marker="o",
    linewidth=2,
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Expected Non-Identity Logical Events per Inference"
)

plt.title(
    "Accumulated Logical-Fault Burden of the Final FT Model"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


EVENT_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_final_ft_expected_logical_events.png",
)


plt.savefig(
    EVENT_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# SAVE GADGET TABLE
# ============================================================

GADGET_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_final_ft_gadget_rates.csv",
)


with open(
    GADGET_CSV,
    "w",
    newline="",
) as file:

    fieldnames = list(
        gadget_rows[0].keys()
    )


    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )


    writer.writeheader()


    writer.writerows(
        gadget_rows
    )


# ============================================================
# FINAL RESOURCE / ASSUMPTION REPORT
# ============================================================

RESOURCE_REPORT = os.path.join(
    RESULT_DIR,
    "cifar10_final_ft_resource_report.txt",
)


with open(
    RESOURCE_REPORT,
    "w",
) as file:

    file.write(
        "FINAL END-TO-END FT CIFAR-10 RESOURCE REPORT\n"
    )

    file.write(
        "==========================================\n\n"
    )

    file.write(
        f"Selected decoder: "
        f"{FINAL_DECODER}\n"
    )

    file.write(
        f"Logical qubits: "
        f"{LOGICAL_QUBITS}\n"
    )

    file.write(
        f"[[5,1,3]] physical data qubits: "
        f"{PHYSICAL_DATA_QUBITS}\n"
    )

    file.write(
        f"Blockwise syndrome/flag ancilla estimate: "
        f"{BLOCKWISE_ANCILLA_ESTIMATE}\n"
    )

    file.write(
        f"Conceptual parallel qubit estimate: "
        f"{CONCEPTUAL_PARALLEL_QUBITS}\n\n"
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
        f"{LOGICAL_CNOT_COUNT}\n"
    )

    file.write(
        f"Uniform logical locations: "
        f"{UNIFORM_LOGICAL_LOCATIONS}\n"
    )

    file.write(
        f"Rotation gadget cycles: "
        f"{ROTATION_GADGET_CYCLES}\n"
    )

    file.write(
        f"CNOT gadget cycles: "
        f"{CNOT_GADGET_CYCLES}\n"
    )

    file.write(
        f"FT-equivalent protected cycles: "
        f"{FT_EQUIVALENT_CYCLES}\n\n"
    )

    file.write(
        f"Ideal classifier accuracy: "
        f"{ideal_accuracy:.6f}\n"
    )

    file.write(
        f"Target accuracy: "
        f"{TARGET_ACCURACY:.6f}\n"
    )

    file.write(
        f"Max tested unprotected p meeting target: "
        f"{max_unprotected_77}\n"
    )

    file.write(
        f"Max tested FinalFT p meeting target: "
        f"{max_final_ft_77}\n"
    )

    file.write(
        f"Approximate FinalFT/unprotected break-even p: "
        f"{break_even_p}\n\n"
    )

    file.write(
        "SCIENTIFIC LIMITATIONS\n"
    )

    file.write(
        "----------------------\n"
    )

    file.write(
        "1. The 8-qubit statevector represents logical qubits; "
        "a 40+ qubit encoded statevector is not explicitly "
        "simulated.\n"
    )

    file.write(
        "2. The selected decoder output is available only as "
        "total p_L, so the residual logical channel is "
        "symmetrized as X=Y=Z=p_L/3.\n"
    )

    file.write(
        "3. Arbitrary logical RY/RZ operations are represented "
        "by FT-overhead cycles rather than explicit compiled "
        "Clifford+T/injection circuits.\n"
    )

    file.write(
        "4. Logical CNOT participant noise is modeled "
        "independently; correlated two-logical-block errors "
        "from an explicit pieceable CNOT gadget are not yet "
        "simulated.\n"
    )

    file.write(
        "5. Qubit estimates describe blockwise EC and do not "
        "claim an exact hardware ancilla count for the final "
        "Rigetti compilation.\n"
    )


# ============================================================
# FINAL FILE SUMMARY
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
    "Per-class CSV:",
    PER_CLASS_CSV,
)

print(
    "Gadget CSV:",
    GADGET_CSV,
)

print(
    "Resource report:",
    RESOURCE_REPORT,
)

print(
    "Final accuracy figure:",
    FINAL_ACCURACY_FIGURE,
)

print(
    "Gain figure:",
    GAIN_FIGURE,
)

print(
    "Overhead figure:",
    OVERHEAD_FIGURE,
)

print(
    "Gadget-error figure:",
    GADGET_FIGURE,
)

print(
    "Expected-events figure:",
    EVENT_FIGURE,
)


print("\n======================================")
print("FINAL INTERPRETATION")
print("======================================")

print(
    "This script freezes one fixed syndrome-decoder policy "
    "for the entire physical-noise sweep."
)

print(
    "UniformProtected applies the selected protected logical "
    "channel once at every modeled circuit location."
)

print(
    "FinalFT adds gate-aware overhead: arbitrary RY/RZ "
    "rotations and logical CNOT participants each incur the "
    "configured number of protected EC-equivalent cycles."
)

print(
    "The final result therefore combines the CIFAR-10 model, "
    "general Pauli noise, [[5,1,3]] protection, noisy/flagged "
    "syndrome information, decoder performance, and logical-"
    "gate overhead in one end-to-end logical simulation."
)

print(
    "This is the final simulator-stage model for the report."
)

print(
    "The next stage is not another FT toy simulation: it is "
    "hardware preparation, where the selected inference "
    "circuits are compiled and validated for Amazon Braket / "
    "Rigetti with finite shots and hardware-specific topology."
)
