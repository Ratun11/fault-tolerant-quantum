import os
import csv
import time
import random
import itertools

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

N_TRAJECTORIES = 32
N_SEEDS = 5


P_VALUES = [
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
# LOAD DATA
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
# FIVE-QUBIT PERFECT CODE
#
# [[5,1,3]]
#
# Stabilizer generators:
#
# X Z Z X I
# I X Z Z X
# X I X Z Z
# Z X I X Z
#
# Logical X = X X X X X
# Logical Z = Z Z Z Z Z
# ============================================================

STABILIZER_STRINGS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ"
]

LOGICAL_X_STRING = "XXXXX"
LOGICAL_Z_STRING = "ZZZZZ"


# Pauli codes:
#
# 0 = I
# 1 = X
# 2 = Y
# 3 = Z

PAULI_CHARS = [
    "I",
    "X",
    "Y",
    "Z"
]


def pauli_string_to_symplectic(
    pauli_string
):

    x = np.zeros(
        len(pauli_string),
        dtype=np.uint8
    )

    z = np.zeros(
        len(pauli_string),
        dtype=np.uint8
    )


    for i, p in enumerate(
        pauli_string
    ):

        if p == "X":

            x[i] = 1

        elif p == "Y":

            x[i] = 1
            z[i] = 1

        elif p == "Z":

            z[i] = 1


    return (
        x,
        z
    )


STABILIZERS = [

    pauli_string_to_symplectic(
        s
    )

    for s in STABILIZER_STRINGS
]


LOGICAL_X = (
    pauli_string_to_symplectic(
        LOGICAL_X_STRING
    )
)


LOGICAL_Z = (
    pauli_string_to_symplectic(
        LOGICAL_Z_STRING
    )
)


ZERO_PAULI = (

    np.zeros(
        5,
        dtype=np.uint8
    ),

    np.zeros(
        5,
        dtype=np.uint8
    )
)


# ============================================================
# SYMPLECTIC COMMUTATION
#
# Returns:
#
# 0 -> commute
# 1 -> anticommute
# ============================================================

def symplectic_inner(
    p1,
    p2
):

    x1, z1 = p1
    x2, z2 = p2


    value = (

        int(
            np.dot(
                x1,
                z2
            )
        )

        +

        int(
            np.dot(
                z1,
                x2
            )
        )

    ) % 2


    return value


# ============================================================
# STABILIZER SYNDROME
# ============================================================

def syndrome(
    pauli
):

    return tuple(

        symplectic_inner(
            pauli,
            stabilizer
        )

        for stabilizer in STABILIZERS
    )


# ============================================================
# PAULI MULTIPLICATION
#
# Global phase can be ignored.
# ============================================================

def multiply_paulis(
    p1,
    p2
):

    return (

        np.bitwise_xor(
            p1[0],
            p2[0]
        ),

        np.bitwise_xor(
            p1[1],
            p2[1]
        )
    )


# ============================================================
# BUILD MINIMUM-WEIGHT DECODER
#
# Every nonzero syndrome corresponds uniquely to one
# single-qubit X, Y, or Z error.
# ============================================================

DECODER = {}


for qubit in range(
    5
):

    for pauli_char in [
        "X",
        "Y",
        "Z"
    ]:

        error = [
            "I"
        ] * 5

        error[
            qubit
        ] = pauli_char


        error = (
            pauli_string_to_symplectic(
                "".join(error)
            )
        )


        error_syndrome = (
            syndrome(
                error
            )
        )


        DECODER[
            error_syndrome
        ] = error


print("\n======================================")
print("5-QUBIT CODE")
print("======================================")

print(
    "Decoder syndromes:",
    len(DECODER)
)

assert len(
    DECODER
) == 15


# ============================================================
# CONVERT FIVE PHYSICAL PAULIS
# INTO RESIDUAL LOGICAL PAULI
#
# output:
#
# 0 = logical I
# 1 = logical X
# 2 = logical Y
# 3 = logical Z
# ============================================================

def physical_pattern_to_logical(
    pattern
):

    pauli_string = "".join(

        PAULI_CHARS[
            code
        ]

        for code in pattern
    )


    physical_error = (
        pauli_string_to_symplectic(
            pauli_string
        )
    )


    error_syndrome = (
        syndrome(
            physical_error
        )
    )


    if error_syndrome == (
        0,
        0,
        0,
        0
    ):

        correction = (
            ZERO_PAULI
        )

    else:

        correction = (
            DECODER[
                error_syndrome
            ]
        )


    residual = multiply_paulis(

        correction,

        physical_error
    )


    # After correction residual must commute
    # with every stabilizer.

    assert syndrome(
        residual
    ) == (
        0,
        0,
        0,
        0
    )


    # Logical X component:
    #
    # Logical X anticommutes with logical Z.

    logical_x_bit = (
        symplectic_inner(
            residual,
            LOGICAL_Z
        )
    )


    # Logical Z component:
    #
    # Logical Z anticommutes with logical X.

    logical_z_bit = (
        symplectic_inner(
            residual,
            LOGICAL_X
        )
    )


    if (
        logical_x_bit == 0
        and
        logical_z_bit == 0
    ):

        return 0


    if (
        logical_x_bit == 1
        and
        logical_z_bit == 0
    ):

        return 1


    if (
        logical_x_bit == 1
        and
        logical_z_bit == 1
    ):

        return 2


    return 3


# ============================================================
# PRECOMPUTE ALL 4^5 = 1024 ERROR PATTERNS
# ============================================================

PATTERN_WEIGHTS = []
PATTERN_LOGICAL_CLASSES = []


for pattern in itertools.product(

    range(4),

    repeat=5
):

    weight = sum(

        code != 0

        for code in pattern
    )


    logical_class = (
        physical_pattern_to_logical(
            pattern
        )
    )


    PATTERN_WEIGHTS.append(
        weight
    )

    PATTERN_LOGICAL_CLASSES.append(
        logical_class
    )


PATTERN_WEIGHTS = np.array(
    PATTERN_WEIGHTS
)

PATTERN_LOGICAL_CLASSES = np.array(
    PATTERN_LOGICAL_CLASSES
)


print(
    "Enumerated physical Pauli patterns:",
    len(PATTERN_WEIGHTS)
)


# ============================================================
# EXACT LOGICAL PAULI DISTRIBUTION
#
# Physical depolarizing:
#
# I : 1-p
#
# X : p/3
# Y : p/3
# Z : p/3
#
#
# Returns:
#
# [P(I_L), P(X_L), P(Y_L), P(Z_L)]
# ============================================================

def five_qubit_logical_distribution(
    p
):

    distribution = np.zeros(
        4,
        dtype=np.float64
    )


    for weight, logical_class in zip(

        PATTERN_WEIGHTS,

        PATTERN_LOGICAL_CLASSES
    ):

        probability = (

            (1.0 - p)
            ** (
                5 - weight
            )

            *

            (p / 3.0)
            ** weight
        )


        distribution[
            logical_class
        ] += probability


    # Numerical normalization

    distribution = (

        distribution

        /

        distribution.sum()
    )


    return distribution


# ============================================================
# VERIFY SINGLE PHYSICAL ERRORS ARE CORRECTED
# ============================================================

for qubit in range(
    5
):

    for error_code in [
        1,
        2,
        3
    ]:

        pattern = [
            0
        ] * 5

        pattern[
            qubit
        ] = error_code


        assert (
            physical_pattern_to_logical(
                pattern
            )
            == 0
        )


print(
    "Single-qubit X/Y/Z correction verified."
)


# ============================================================
# PRINT EXACT LOGICAL ERROR RATES
# ============================================================

print("\n======================================")
print("EXACT [[5,1,3]] LOGICAL DISTRIBUTIONS")
print("======================================")


LOGICAL_DISTRIBUTIONS = {}


for p in P_VALUES:

    distribution = (
        five_qubit_logical_distribution(
            p
        )
    )


    LOGICAL_DISTRIBUTIONS[
        p
    ] = distribution


    p_logical = (
        1.0
        -
        distribution[0]
    )


    print(

        f"p={p:.4f} | "

        f"I={distribution[0]:.8f} | "

        f"X={distribution[1]:.8f} | "

        f"Y={distribution[2]:.8f} | "

        f"Z={distribution[3]:.8f} | "

        f"pL={p_logical:.8f}"
    )


# ============================================================
# FAST GPU STATEVECTOR
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


    if not torch.is_tensor(
        theta
    ):

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


    if not torch.is_tensor(
        theta
    ):

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
        -0.5j
        * theta
    )

    phase1 = torch.exp(
        0.5j
        * theta
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
# X
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


    a = x[
        :, :, 0, :
    ]

    b = x[
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
# Z
# ============================================================

def apply_z_fault(
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


    a = x[
        :, :, 0, :
    ]

    b = x[
        :, :, 1, :
    ]


    mask = mask.bool().reshape(
        batch_size,
        1,
        1
    )


    out0 = a


    out1 = torch.where(
        mask,
        -b,
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
# PAULI FAULT
#
# 0 = I
# 1 = X
# 2 = Y
# 3 = Z
#
# Y can be represented as XZ up to a global phase,
# which does not affect these measurement statistics.
# ============================================================

def apply_pauli_fault(
    state,
    pauli_codes,
    wire
):

    x_mask = torch.logical_or(

        pauli_codes == 1,

        pauli_codes == 2
    )


    z_mask = torch.logical_or(

        pauli_codes == 2,

        pauli_codes == 3
    )


    state = apply_z_fault(

        state,

        z_mask,

        wire
    )


    state = apply_x_fault(

        state,

        x_mask,

        wire
    )


    return state


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
            target
        )
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
            (
                control,
                target
            )
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
# SAMPLE PAULI ERRORS
#
# probabilities:
#
# [I, X, Y, Z]
# ============================================================

def sample_pauli_codes(
    shape,
    probabilities
):

    p_i = float(
        probabilities[0]
    )

    p_x = float(
        probabilities[1]
    )

    p_y = float(
        probabilities[2]
    )


    random_values = torch.rand(

        shape,

        device=device
    )


    codes = torch.zeros(

        shape,

        dtype=torch.uint8,

        device=device
    )


    codes[
        random_values >= p_i
    ] = 1


    codes[
        random_values
        >= (
            p_i
            +
            p_x
        )
    ] = 2


    codes[
        random_values
        >= (
            p_i
            +
            p_x
            +
            p_y
        )
    ] = 3


    return codes


# ============================================================
# ERROR DISTRIBUTION
#
# unprotected:
#
# I = 1-p
# X = Y = Z = p/3
#
#
# code5:
#
# exact residual logical distribution after
# ideal [[5,1,3]] syndrome decoding.
# ============================================================

def get_error_distribution(
    p,
    mode
):

    if mode == "none":

        return np.array(
            [
                1.0 - p,
                p / 3.0,
                p / 3.0,
                p / 3.0
            ],
            dtype=np.float64
        )


    if mode == "code5":

        return (
            five_qubit_logical_distribution(
                p
            )
        )


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

def run_quantum_circuit(
    inputs,
    quantum_weights,
    p,
    mode
):

    batch_size = (
        inputs.shape[0]
    )


    state = initial_state(
        batch_size
    )


    error_distribution = (
        get_error_distribution(
            p,
            mode
        )
    )


    # ========================================================
    # SAMPLE PAULI ERRORS
    # ========================================================

    encoding_errors = sample_pauli_codes(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        error_distribution
    )


    ry_errors = sample_pauli_codes(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        error_distribution
    )


    rz_errors = sample_pauli_codes(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS
        ),

        error_distribution
    )


    cnot_errors = sample_pauli_codes(

        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2
        ),

        error_distribution
    )


    # ========================================================
    # VQC
    # ========================================================

    for upload in range(
        N_UPLOADS
    ):

        start = (
            upload
            *
            N_QUBITS
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


            state = apply_pauli_fault(

                state,

                encoding_errors[
                    :,
                    upload,
                    q
                ],

                q
            )


        # ----------------------------------------------------
        # TRAINABLE GATES
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


            state = apply_pauli_fault(

                state,

                ry_errors[
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


            state = apply_pauli_fault(

                state,

                rz_errors[
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

                target
            )


            state = apply_pauli_fault(

                state,

                cnot_errors[
                    :,
                    upload,
                    gate_index,
                    0
                ],

                control
            )


            state = apply_pauli_fault(

                state,

                cnot_errors[
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


    return (

        probabilities

        @

        Z_SIGN_MATRIX
    )


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier(
    p,
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


            if p == 0:

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

                    p=p,

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


(
    ideal_accuracy,
    ideal_f1,
    ideal_time

) = evaluate_classifier(

    p=0.0,

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
print("DEPOLARIZING-NOISE EXPERIMENT")
print("======================================")

print(
    "Trajectories:",
    N_TRAJECTORIES
)

print(
    "Seeds:",
    N_SEEDS
)


detailed_results = []


for p_index, p in enumerate(
    P_VALUES
):

    logical_distribution = (
        five_qubit_logical_distribution(
            p
        )
    )


    logical_error = (
        1.0
        -
        logical_distribution[0]
    )


    print(
        "\n======================================"
    )

    print(
        f"Physical depolarizing p = "
        f"{p:.4f}"
    )

    print(
        f"[[5,1,3]] logical error = "
        f"{logical_error:.8f}"
    )

    print(
        "Logical distribution:"
    )

    print(

        f"I={logical_distribution[0]:.8f}, "

        f"X={logical_distribution[1]:.8f}, "

        f"Y={logical_distribution[2]:.8f}, "

        f"Z={logical_distribution[3]:.8f}"
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

            p_index * 1000

            +

            seed_index
        )


        if p == 0:

            acc_none = (
                ideal_accuracy
            )

            f1_none = (
                ideal_f1
            )

            time_none = 0.0


            acc_code = (
                ideal_accuracy
            )

            f1_code = (
                ideal_f1
            )

            time_code = 0.0


        else:

            (
                acc_none,
                f1_none,
                time_none

            ) = evaluate_classifier(

                p=p,

                mode="none",

                random_seed=(
                    base_seed
                )
            )


            (
                acc_code,
                f1_code,
                time_code

            ) = evaluate_classifier(

                p=p,

                mode="code5",

                random_seed=(
                    base_seed
                    +
                    100000
                )
            )


        detailed_results.append({

            "p":
                p,

            "seed":
                base_seed,

            "logical_I":
                logical_distribution[0],

            "logical_X":
                logical_distribution[1],

            "logical_Y":
                logical_distribution[2],

            "logical_Z":
                logical_distribution[3],

            "logical_error":
                logical_error,

            "unprotected_accuracy":
                acc_none,

            "code5_accuracy":
                acc_code,

            "unprotected_f1":
                f1_none,

            "code5_f1":
                f1_code,

            "unprotected_seconds":
                time_none,

            "code5_seconds":
                time_code
        })


        print(

            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "

            f"Unprotected="
            f"{acc_none:.4f} | "

            f"[[5,1,3]]="
            f"{acc_code:.4f}"
        )


# ============================================================
# DETAILED CSV
# ============================================================

DETAILED_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_depolarizing_5qubit_detailed.csv"
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

        "physical_depolarizing_p",

        "seed",

        "logical_I_probability",

        "logical_X_probability",

        "logical_Y_probability",

        "logical_Z_probability",

        "logical_error_probability",

        "unprotected_accuracy",

        "five_qubit_accuracy",

        "unprotected_macro_f1",

        "five_qubit_macro_f1",

        "unprotected_seconds",

        "five_qubit_seconds"
    ])


    for row in detailed_results:

        writer.writerow([

            row["p"],

            row["seed"],

            row["logical_I"],

            row["logical_X"],

            row["logical_Y"],

            row["logical_Z"],

            row["logical_error"],

            row["unprotected_accuracy"],

            row["code5_accuracy"],

            row["unprotected_f1"],

            row["code5_f1"],

            row["unprotected_seconds"],

            row["code5_seconds"]
        ])


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


    none_acc = np.array([

        row["unprotected_accuracy"]

        for row in subset
    ])


    code_acc = np.array([

        row["code5_accuracy"]

        for row in subset
    ])


    none_f1 = np.array([

        row["unprotected_f1"]

        for row in subset
    ])


    code_f1 = np.array([

        row["code5_f1"]

        for row in subset
    ])


    summary.append({

        "p":
            p,

        "logical_error":
            subset[0][
                "logical_error"
            ],

        "logical_X":
            subset[0][
                "logical_X"
            ],

        "logical_Y":
            subset[0][
                "logical_Y"
            ],

        "logical_Z":
            subset[0][
                "logical_Z"
            ],

        "none_mean":
            none_acc.mean(),

        "none_std":
            none_acc.std(
                ddof=1
            )
            if len(none_acc) > 1
            else 0.0,

        "code_mean":
            code_acc.mean(),

        "code_std":
            code_acc.std(
                ddof=1
            )
            if len(code_acc) > 1
            else 0.0,

        "none_f1":
            none_f1.mean(),

        "code_f1":
            code_f1.mean()
    })


# ============================================================
# SUMMARY CSV
# ============================================================

SUMMARY_CSV = os.path.join(

    RESULT_DIR,

    "cifar10_depolarizing_5qubit_summary.csv"
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

        "physical_depolarizing_p",

        "logical_error_probability",

        "logical_X_probability",

        "logical_Y_probability",

        "logical_Z_probability",

        "unprotected_accuracy_mean",

        "unprotected_accuracy_std",

        "five_qubit_accuracy_mean",

        "five_qubit_accuracy_std",

        "unprotected_macro_f1",

        "five_qubit_macro_f1"
    ])


    for row in summary:

        writer.writerow([

            row["p"],

            row["logical_error"],

            row["logical_X"],

            row["logical_Y"],

            row["logical_Z"],

            row["none_mean"],

            row["none_std"],

            row["code_mean"],

            row["code_std"],

            row["none_f1"],

            row["code_f1"]
        ])


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p\t\t"
    "p_L\t\t"
    "Unprotected\t"
    "[[5,1,3]]"
)


for row in summary:

    print(

        f"{row['p']:.4f}\t\t"

        f"{row['logical_error']:.6f}\t"

        f"{row['none_mean']:.4f}"
        f" ± "
        f"{row['none_std']:.4f}\t"

        f"{row['code_mean']:.4f}"
        f" ± "
        f"{row['code_std']:.4f}"
    )


# ============================================================
# PLOT ARRAYS
# ============================================================

p_values = np.array([

    row["p"]

    for row in summary
])


logical_error = np.array([

    row["logical_error"]

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


code_mean = np.array([

    row["code_mean"]

    for row in summary
])


code_std = np.array([

    row["code_std"]

    for row in summary
])


logical_x = np.array([

    row["logical_X"]

    for row in summary
])


logical_y = np.array([

    row["logical_Y"]

    for row in summary
])


logical_z = np.array([

    row["logical_Z"]

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

    label="Unprotected depolarizing noise"
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

    code_mean * 100,

    marker="s",

    linewidth=2,

    label="Ideal [[5,1,3]] QEC proxy"
)


plt.fill_between(

    p_values * 100,

    (
        code_mean
        -
        code_std
    ) * 100,

    (
        code_mean
        +
        code_std
    ) * 100,

    alpha=0.15
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "CIFAR-10 Under General Pauli Noise"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_depolarizing_5qubit_accuracy.png"
)


plt.savefig(

    ACCURACY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 2
# PHYSICAL VS LOGICAL ERROR
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    p_values * 100,

    marker="o",

    linewidth=2,

    label="Physical depolarizing p"
)


plt.plot(

    p_values * 100,

    logical_error * 100,

    marker="s",

    linewidth=2,

    label="[[5,1,3]] residual logical p"
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Error Probability (%)"
)

plt.title(
    "[[5,1,3]] General-Pauli Error Suppression"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ERROR_FIGURE = os.path.join(

    FIGURE_DIR,

    "five_qubit_physical_vs_logical_depolarizing.png"
)


plt.savefig(

    ERROR_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 3
# RESIDUAL LOGICAL PAULI DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(

    p_values * 100,

    logical_x * 100,

    marker="o",

    linewidth=2,

    label="Logical X"
)


plt.plot(

    p_values * 100,

    logical_y * 100,

    marker="s",

    linewidth=2,

    label="Logical Y"
)


plt.plot(

    p_values * 100,

    logical_z * 100,

    marker="^",

    linewidth=2,

    label="Logical Z"
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Residual Logical Pauli Probability (%)"
)

plt.title(
    "Residual Logical Errors After [[5,1,3]] QEC"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


PAULI_FIGURE = os.path.join(

    FIGURE_DIR,

    "five_qubit_logical_pauli_distribution.png"
)


plt.savefig(

    PAULI_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FIGURE 4
# ACCURACY RECOVERY
# ============================================================

recovery = (

    code_mean

    -

    none_mean
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
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Accuracy Recovery (percentage points)"
)

plt.title(
    "Accuracy Recovered by [[5,1,3]] QEC"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


RECOVERY_FIGURE = os.path.join(

    FIGURE_DIR,

    "cifar10_5qubit_accuracy_recovery.png"
)


plt.savefig(

    RECOVERY_FIGURE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# FILES
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
    "Error figure:",
    ERROR_FIGURE
)

print(
    "Logical Pauli figure:",
    PAULI_FIGURE
)

print(
    "Recovery figure:",
    RECOVERY_FIGURE
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "The physical channel now contains "
    "X, Y, and Z errors."
)

print(
    "The [[5,1,3]] code corrects any "
    "single-qubit Pauli error."
)

print(
    "All 4^5 physical Pauli patterns are "
    "enumerated exactly to obtain the "
    "residual logical I/X/Y/Z distribution."
)

print(
    "The 8-qubit classifier is still simulated "
    "as logical qubits for efficiency."
)

print(
    "Encoding, syndrome extraction, recovery, "
    "and arbitrary logical RY/RZ gates are "
    "assumed ideal in this experiment."
)

print(
    "Therefore this is an ideal five-qubit-code "
    "QEC proxy under depolarizing noise, not yet "
    "a circuit-level fault-tolerant implementation."
)
