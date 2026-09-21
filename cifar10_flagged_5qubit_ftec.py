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
# FLAGGED [[5,1,3]] FAULT-TOLERANT ERROR-CORRECTION STUDY
#
# This script is the next step after the naive noisy
# stabilizer-extraction experiment.
#
# It compares:
#
#   1) Unprotected depolarizing noise
#   2) Ideal [[5,1,3]] QEC
#   3) Naive noisy one-round stabilizer QEC
#   4) Flagged [[5,1,3]] FTEC
#
# The flagged procedure follows the two-ancilla logic of
# Chao & Reichardt:
#
#   - one syndrome ancilla
#   - one flag ancilla
#   - if a flag fires OR the measured stabilizer syndrome
#     is nontrivial, extract all four syndromes and decode
#   - when a flag fires, use a flag-aware correction table
#     that distinguishes correlated hook errors
#
# The five physical data qubits are tracked as Pauli frames.
# The 8-qubit CIFAR-10 VQC remains a logical statevector.
#
# IMPORTANT:
# This is still not a fully encoded implementation of
# arbitrary logical RY/RZ gates. It specifically upgrades
# the error-correction gadget to a flag-aware FTEC model.
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

# Monte-Carlo encoded-block samples used to estimate one
# logical Pauli channel.
BLOCK_SAMPLES = 500_000


# ============================================================
# PHYSICAL DATA NOISE
# ============================================================

P_VALUES = [
    0.0,
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050,
]


# ============================================================
# ERROR-CORRECTION CIRCUIT NOISE
# ============================================================

# Two-qubit depolarizing probability after each syndrome
# extraction CNOT.
P_2Q = 0.005

# One-qubit depolarizing probability after each basis-change H.
P_1Q = 0.001

# Depolarizing probability associated with ancilla preparation.
#
# For a freshly prepared |0>, Z is physically harmless and
# Y is equivalent to X up to phase.
#
# For a freshly prepared |+>, X is harmless and Y is
# equivalent to Z up to phase.
#
# The code below handles these prepared-state equivalences
# explicitly instead of treating every Pauli as distinct.
P_PREP = 0.001

# Classical ancilla measurement-bit error probability.
P_MEAS = 0.010

# Error following a physically executed Pauli recovery.
P_RECOVERY = 0.005

# Keep False for direct continuity with the preceding
# experiment, where recovery itself was noisy.
#
# If True, corrections are treated as a classical Pauli frame
# update and P_RECOVERY is not applied.
USE_PAULI_FRAME_RECOVERY = False


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "results/cifar10_resnet32_features.npz"
MODEL_PATH = "results/cifar10_quantum_batched_best.pt"

RESULT_DIR = "results"
FIGURE_DIR = "figures"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


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
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)


# ============================================================
# LOAD TEST DATA
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True,
)

X_test = data["X_test"].astype(
    np.float32
)

y_test = data["y_test"].astype(
    np.int64
)


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


print("\n======================================")
print("TEST DATA")
print("======================================")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


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
print("Loaded:", MODEL_PATH)
print(
    "Quantum weights:",
    model.quantum_weights.device,
)
print(
    "Classifier:",
    model.classifier[0].weight.device,
)


# ============================================================
# [[5,1,3]] PERFECT CODE
# ============================================================

STABILIZER_STRINGS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]

LOGICAL_X_STRING = "XXXXX"
LOGICAL_Z_STRING = "ZZZZZ"

PAULI_CHARS = [
    "I",
    "X",
    "Y",
    "Z",
]


def pauli_string_to_symplectic(
    pauli_string,
):

    x = np.zeros(
        len(pauli_string),
        dtype=np.uint8,
    )

    z = np.zeros(
        len(pauli_string),
        dtype=np.uint8,
    )

    for i, p in enumerate(
        pauli_string
    ):

        if p in (
            "X",
            "Y",
        ):
            x[i] = 1

        if p in (
            "Z",
            "Y",
        ):
            z[i] = 1

    return x, z


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


def symplectic_inner_numpy(
    p1,
    p2,
):

    x1, z1 = p1
    x2, z2 = p2

    return int(
        (
            np.dot(
                x1,
                z2,
            )
            +
            np.dot(
                z1,
                x2,
            )
        )
        % 2
    )


def syndrome_numpy(
    pauli,
):

    return tuple(
        symplectic_inner_numpy(
            pauli,
            stabilizer,
        )
        for stabilizer in STABILIZERS
    )


def syndrome_tuple_to_index(
    s,
):

    return (
        8 * int(s[0])
        +
        4 * int(s[1])
        +
        2 * int(s[2])
        +
        int(s[3])
    )


def multiply_paulis_numpy(
    p1,
    p2,
):

    return (
        np.bitwise_xor(
            p1[0],
            p2[0],
        ),
        np.bitwise_xor(
            p1[1],
            p2[1],
        ),
    )


def symplectic_to_string(
    x,
    z,
):

    output = []

    for xb, zb in zip(
        x,
        z,
    ):

        if not xb and not zb:
            output.append("I")

        elif xb and not zb:
            output.append("X")

        elif xb and zb:
            output.append("Y")

        else:
            output.append("Z")

    return "".join(
        output
    )


# ============================================================
# STANDARD MINIMUM-WEIGHT DECODER
# ============================================================

DECODER = {}

for qubit in range(5):

    for pauli_char in [
        "X",
        "Y",
        "Z",
    ]:

        error = ["I"] * 5
        error[qubit] = pauli_char

        error = (
            pauli_string_to_symplectic(
                "".join(error)
            )
        )

        DECODER[
            syndrome_numpy(
                error
            )
        ] = error


assert len(DECODER) == 15


DECODER_X_NP = np.zeros(
    (
        16,
        5,
    ),
    dtype=np.uint8,
)

DECODER_Z_NP = np.zeros(
    (
        16,
        5,
    ),
    dtype=np.uint8,
)


for syn, correction in DECODER.items():

    idx = syndrome_tuple_to_index(
        syn
    )

    DECODER_X_NP[
        idx
    ] = correction[0]

    DECODER_Z_NP[
        idx
    ] = correction[1]


DECODER_X = torch.tensor(
    DECODER_X_NP,
    dtype=torch.bool,
    device=device,
)

DECODER_Z = torch.tensor(
    DECODER_Z_NP,
    dtype=torch.bool,
    device=device,
)


STAB_X = torch.tensor(
    np.stack(
        [
            s[0]
            for s in STABILIZERS
        ]
    ),
    dtype=torch.bool,
    device=device,
)

STAB_Z = torch.tensor(
    np.stack(
        [
            s[1]
            for s in STABILIZERS
        ]
    ),
    dtype=torch.bool,
    device=device,
)


LOGICAL_X_X = torch.tensor(
    LOGICAL_X[0],
    dtype=torch.bool,
    device=device,
)

LOGICAL_X_Z = torch.tensor(
    LOGICAL_X[1],
    dtype=torch.bool,
    device=device,
)

LOGICAL_Z_X = torch.tensor(
    LOGICAL_Z[0],
    dtype=torch.bool,
    device=device,
)

LOGICAL_Z_Z = torch.tensor(
    LOGICAL_Z[1],
    dtype=torch.bool,
    device=device,
)


# ============================================================
# IDEAL FLAG-CIRCUIT SINGLE-FAULT ENUMERATION
#
# The flagged weight-4 stabilizer circuit uses:
#
#   data interaction 0
#   flag -> syndrome
#   data interaction 1
#   data interaction 2
#   flag -> syndrome
#   data interaction 3
#
# Syndrome ancilla: |0>, measured in Z.
# Flag ancilla:     |+>, measured in X.
#
# A Z/Y component on the syndrome ancilla that can propagate
# into a dangerous correlated data error is copied to the
# flag ancilla and becomes visible in its X measurement.
#
# We enumerate every non-II two-qubit Pauli fault after every
# two-qubit gate in the ideal flagged circuit.
#
# This automatically constructs the flag-aware decoder table.
# ============================================================

TWO_Q_PAULI_PAIRS = [
    a + b
    for a in "IXYZ"
    for b in "IXYZ"
    if a + b != "II"
]


def char_to_xz(
    p,
):

    return (
        1 if p in (
            "X",
            "Y",
        ) else 0,
        1 if p in (
            "Z",
            "Y",
        ) else 0,
    )


def cnot_frame_scalar(
    cx,
    cz,
    tx,
    tz,
):

    # X_control propagates to target.
    # Z_target propagates to control.

    tx_new = (
        tx ^ cx
    )

    cz_new = (
        cz ^ tz
    )

    return (
        cx,
        cz_new,
        tx_new,
        tz,
    )


def simulate_single_fault_flagged_circuit(
    stabilizer_string,
    fault_location,
    fault_pair,
):

    dx = np.zeros(
        5,
        dtype=np.uint8,
    )

    dz = np.zeros(
        5,
        dtype=np.uint8,
    )

    sx = 0
    sz = 0

    fx = 0
    fz = 0


    support = [
        q
        for q, p in enumerate(
            stabilizer_string
        )
        if p != "I"
    ]


    def inject_pair(
        first_kind,
        first_index,
        pair,
    ):

        nonlocal sx
        nonlocal sz
        nonlocal fx
        nonlocal fz
        nonlocal dx
        nonlocal dz

        p1 = pair[0]
        p2 = pair[1]

        x1, z1 = char_to_xz(
            p1
        )

        x2, z2 = char_to_xz(
            p2
        )


        if first_kind == "data":

            q = support[
                first_index
            ]

            dx[q] ^= x1
            dz[q] ^= z1

            sx ^= x2
            sz ^= z2

        elif first_kind == "flag":

            fx ^= x1
            fz ^= z1

            sx ^= x2
            sz ^= z2

        else:

            raise ValueError(
                "Unknown fault kind."
            )


    for k, q in enumerate(
        support
    ):

        component = (
            stabilizer_string[q]
        )


        # Measure X component by basis-changing the data qubit.
        if component == "X":

            dx[q], dz[q] = (
                dz[q],
                dx[q],
            )


        (
            dx[q],
            dz[q],
            sx,
            sz,
        ) = cnot_frame_scalar(
            dx[q],
            dz[q],
            sx,
            sz,
        )


        if fault_location == (
            "data",
            k,
        ):

            inject_pair(
                "data",
                k,
                fault_pair,
            )


        if component == "X":

            dx[q], dz[q] = (
                dz[q],
                dx[q],
            )


        # First flag coupling after first data interaction.
        if k == 0:

            (
                fx,
                fz,
                sx,
                sz,
            ) = cnot_frame_scalar(
                fx,
                fz,
                sx,
                sz,
            )


            if fault_location == (
                "flag",
                0,
            ):

                inject_pair(
                    "flag",
                    0,
                    fault_pair,
                )


        # Second flag coupling after third data interaction.
        if k == 2:

            (
                fx,
                fz,
                sx,
                sz,
            ) = cnot_frame_scalar(
                fx,
                fz,
                sx,
                sz,
            )


            if fault_location == (
                "flag",
                1,
            ):

                inject_pair(
                    "flag",
                    1,
                    fault_pair,
                )


    syndrome_measurement = int(
        sx
    )

    # X-basis flag measurement flips for a Z/Y frame.
    flag_measurement = int(
        fz
    )


    return (
        (
            dx,
            dz,
        ),
        syndrome_measurement,
        flag_measurement,
    )


# ============================================================
# BUILD FLAG-AWARE CORRECTION LOOKUP
#
# Default every syndrome to the ordinary minimum-weight
# decoder. Then overwrite the syndromes that can arise from a
# flagged single fault with the corresponding correlated error.
#
# The first generator should reproduce the published set:
#
# IIIII
# IIZXI
# IXZXI
# IYZXI
# IZZXI
# IIIXI
# IIXXI
# IIYXI
# ============================================================

FLAG_CORR_X_NP = np.repeat(
    DECODER_X_NP[
        None,
        :,
        :,
    ],
    repeats=4,
    axis=0,
)

FLAG_CORR_Z_NP = np.repeat(
    DECODER_Z_NP[
        None,
        :,
        :,
    ],
    repeats=4,
    axis=0,
)


FLAG_LOOKUP_STRINGS = []


print("\n======================================")
print("FLAG SINGLE-FAULT VERIFICATION")
print("======================================")


for stabilizer_index, stabilizer_string in enumerate(
    STABILIZER_STRINGS
):

    syndrome_to_error = {}


    fault_locations = (
        [
            (
                "data",
                k,
            )
            for k in range(4)
        ]
        +
        [
            (
                "flag",
                0,
            ),
            (
                "flag",
                1,
            ),
        ]
    )


    for fault_location in fault_locations:

        for pair in TWO_Q_PAULI_PAIRS:

            (
                data_error,
                _,
                flag_value,
            ) = (
                simulate_single_fault_flagged_circuit(
                    stabilizer_string,
                    fault_location,
                    pair,
                )
            )


            if flag_value == 0:
                continue


            error_syndrome = (
                syndrome_numpy(
                    data_error
                )
            )


            error_string = (
                symplectic_to_string(
                    data_error[0],
                    data_error[1],
                )
            )


            if error_syndrome in syndrome_to_error:

                previous = (
                    syndrome_to_error[
                        error_syndrome
                    ]
                )

                # Different physical strings with the same syndrome
                # are acceptable only if they are stabilizer-
                # equivalent. For this circuit the enumeration should
                # actually give one canonical string per syndrome.
                if previous != error_string:

                    raise RuntimeError(
                        "Ambiguous flagged single-fault syndrome "
                        f"for {stabilizer_string}: "
                        f"{error_syndrome} -> "
                        f"{previous}, {error_string}"
                    )

            else:

                syndrome_to_error[
                    error_syndrome
                ] = error_string


    if len(
        syndrome_to_error
    ) != 8:

        raise RuntimeError(
            "Expected exactly 8 flag-conditioned syndrome "
            f"classes for {stabilizer_string}, got "
            f"{len(syndrome_to_error)}."
        )


    ordered_strings = []


    for syn, error_string in (
        syndrome_to_error.items()
    ):

        error = (
            pauli_string_to_symplectic(
                error_string
            )
        )

        idx = syndrome_tuple_to_index(
            syn
        )

        FLAG_CORR_X_NP[
            stabilizer_index,
            idx,
        ] = error[0]

        FLAG_CORR_Z_NP[
            stabilizer_index,
            idx,
        ] = error[1]

        ordered_strings.append(
            (
                syn,
                error_string,
            )
        )


    FLAG_LOOKUP_STRINGS.append(
        ordered_strings
    )


    print(
        f"{stabilizer_string}: "
        f"{len(syndrome_to_error)} "
        "unique flagged error classes"
    )


published_first_set = {
    "IIIII",
    "IIZXI",
    "IXZXI",
    "IYZXI",
    "IZZXI",
    "IIIXI",
    "IIXXI",
    "IIYXI",
}


first_generated_set = {
    error_string
    for _, error_string in (
        FLAG_LOOKUP_STRINGS[0]
    )
}


assert (
    first_generated_set
    ==
    published_first_set
)


print(
    "XZZXI flag table matches the expected "
    "8-class single-fault correction set."
)


FLAG_CORR_X = torch.tensor(
    FLAG_CORR_X_NP,
    dtype=torch.bool,
    device=device,
)

FLAG_CORR_Z = torch.tensor(
    FLAG_CORR_Z_NP,
    dtype=torch.bool,
    device=device,
)


# ============================================================
# EXACT IDEAL [[5,1,3]] LOGICAL CHANNEL
# ============================================================

def physical_pattern_to_logical(
    pattern,
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

    syn = syndrome_numpy(
        physical_error
    )


    if syn == (
        0,
        0,
        0,
        0,
    ):

        correction = (
            np.zeros(
                5,
                dtype=np.uint8,
            ),
            np.zeros(
                5,
                dtype=np.uint8,
            ),
        )

    else:

        correction = DECODER[
            syn
        ]


    residual = (
        multiply_paulis_numpy(
            correction,
            physical_error,
        )
    )


    logical_x_bit = (
        symplectic_inner_numpy(
            residual,
            LOGICAL_Z,
        )
    )

    logical_z_bit = (
        symplectic_inner_numpy(
            residual,
            LOGICAL_X,
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


PATTERN_WEIGHTS = []
PATTERN_LOGICAL_CLASSES = []


for pattern in itertools.product(
    range(4),
    repeat=5,
):

    PATTERN_WEIGHTS.append(
        sum(
            code != 0
            for code in pattern
        )
    )

    PATTERN_LOGICAL_CLASSES.append(
        physical_pattern_to_logical(
            pattern
        )
    )


PATTERN_WEIGHTS = np.array(
    PATTERN_WEIGHTS
)

PATTERN_LOGICAL_CLASSES = np.array(
    PATTERN_LOGICAL_CLASSES
)


def ideal_five_qubit_logical_distribution(
    p,
):

    distribution = np.zeros(
        4,
        dtype=np.float64,
    )


    for weight, logical_class in zip(
        PATTERN_WEIGHTS,
        PATTERN_LOGICAL_CLASSES,
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


    return (
        distribution
        /
        distribution.sum()
    )


# ============================================================
# RANDOM NOISE HELPERS
# ============================================================

def bernoulli(
    shape,
    p,
):

    if p <= 0:

        return torch.zeros(
            shape,
            dtype=torch.bool,
            device=device,
        )


    return (
        torch.rand(
            shape,
            device=device,
        )
        < p
    )


def sample_single_qubit_depolarizing(
    shape,
    p,
):

    if p <= 0:

        zeros = torch.zeros(
            shape,
            dtype=torch.bool,
            device=device,
        )

        return (
            zeros.clone(),
            zeros.clone(),
        )


    has_error = bernoulli(
        shape,
        p,
    )


    pauli_type = torch.randint(
        low=0,
        high=3,
        size=shape,
        device=device,
    )


    # 0 -> X
    # 1 -> Y
    # 2 -> Z

    x = torch.logical_and(
        has_error,
        pauli_type != 2,
    )

    z = torch.logical_and(
        has_error,
        pauli_type != 0,
    )


    return x, z


def sample_two_qubit_depolarizing(
    shape,
    p,
):

    if p <= 0:

        zeros = torch.zeros(
            shape,
            dtype=torch.bool,
            device=device,
        )

        return (
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
        )


    has_error = bernoulli(
        shape,
        p,
    )


    # 1..15 represents every non-II pair.
    pair_code = torch.randint(
        low=1,
        high=16,
        size=shape,
        device=device,
    )


    first = (
        pair_code
        // 4
    )

    second = (
        pair_code
        % 4
    )


    x1 = torch.logical_and(
        has_error,
        torch.logical_or(
            first == 1,
            first == 2,
        ),
    )

    z1 = torch.logical_and(
        has_error,
        torch.logical_or(
            first == 2,
            first == 3,
        ),
    )


    x2 = torch.logical_and(
        has_error,
        torch.logical_or(
            second == 1,
            second == 2,
        ),
    )

    z2 = torch.logical_and(
        has_error,
        torch.logical_or(
            second == 2,
            second == 3,
        ),
    )


    return (
        x1,
        z1,
        x2,
        z2,
    )


# ============================================================
# ANCILLA PREPARATION NOISE
#
# Depolarizing error after ideal preparation:
#
# |0>:
#   Z|0> = |0>
#   Y|0> = i X|0>
#
# therefore X/Y are the same effective bit error.
#
# |+>:
#   X|+> = |+>
#   Y|+> ~ Z|+>
#
# therefore Z/Y are the same effective phase error.
# ============================================================

def prepare_zero_ancilla(
    samples,
):

    # X or Y occurs with total probability 2 P_PREP / 3.
    sx = bernoulli(
        (samples,),
        2.0 * P_PREP / 3.0,
    )

    sz = torch.zeros(
        samples,
        dtype=torch.bool,
        device=device,
    )

    return sx, sz


def prepare_plus_ancilla(
    samples,
):

    fx = torch.zeros(
        samples,
        dtype=torch.bool,
        device=device,
    )

    # Z or Y occurs with total probability 2 P_PREP / 3.
    fz = bernoulli(
        (samples,),
        2.0 * P_PREP / 3.0,
    )

    return fx, fz


# ============================================================
# PAULI-FRAME GATES FOR SYNDROME EXTRACTION
# ============================================================

def noisy_h_data(
    dx,
    dz,
    q,
):

    old_x = dx[
        :,
        q,
    ].clone()


    dx[
        :,
        q,
    ] = dz[
        :,
        q,
    ]


    dz[
        :,
        q,
    ] = old_x


    fx, fz = (
        sample_single_qubit_depolarizing(
            dx[
                :,
                q,
            ].shape,
            P_1Q,
        )
    )


    dx[
        :,
        q,
    ] = torch.logical_xor(
        dx[
            :,
            q,
        ],
        fx,
    )


    dz[
        :,
        q,
    ] = torch.logical_xor(
        dz[
            :,
            q,
        ],
        fz,
    )


def noisy_data_to_syndrome_cnot(
    dx,
    dz,
    q,
    sx,
    sz,
):

    # data = control
    # syndrome = target

    data_x = dx[
        :,
        q,
    ].clone()

    data_z = dz[
        :,
        q,
    ].clone()

    syn_x = sx.clone()
    syn_z = sz.clone()


    # Ideal CNOT Pauli propagation.
    dx[
        :,
        q,
    ] = data_x

    dz[
        :,
        q,
    ] = torch.logical_xor(
        data_z,
        syn_z,
    )

    sx = torch.logical_xor(
        syn_x,
        data_x,
    )

    sz = syn_z


    (
        fx_data,
        fz_data,
        fx_syn,
        fz_syn,
    ) = sample_two_qubit_depolarizing(
        sx.shape,
        P_2Q,
    )


    dx[
        :,
        q,
    ] = torch.logical_xor(
        dx[
            :,
            q,
        ],
        fx_data,
    )

    dz[
        :,
        q,
    ] = torch.logical_xor(
        dz[
            :,
            q,
        ],
        fz_data,
    )

    sx = torch.logical_xor(
        sx,
        fx_syn,
    )

    sz = torch.logical_xor(
        sz,
        fz_syn,
    )


    return sx, sz


def noisy_flag_to_syndrome_cnot(
    fx,
    fz,
    sx,
    sz,
):

    # flag = control
    # syndrome = target

    flag_x = fx.clone()
    flag_z = fz.clone()

    syn_x = sx.clone()
    syn_z = sz.clone()


    fx = flag_x

    fz = torch.logical_xor(
        flag_z,
        syn_z,
    )

    sx = torch.logical_xor(
        syn_x,
        flag_x,
    )

    sz = syn_z


    (
        fault_x_flag,
        fault_z_flag,
        fault_x_syn,
        fault_z_syn,
    ) = sample_two_qubit_depolarizing(
        sx.shape,
        P_2Q,
    )


    fx = torch.logical_xor(
        fx,
        fault_x_flag,
    )

    fz = torch.logical_xor(
        fz,
        fault_z_flag,
    )

    sx = torch.logical_xor(
        sx,
        fault_x_syn,
    )

    sz = torch.logical_xor(
        sz,
        fault_z_syn,
    )


    return (
        fx,
        fz,
        sx,
        sz,
    )


# ============================================================
# UNFLAGGED STABILIZER MEASUREMENT
# ============================================================

def measure_stabilizer_unflagged(
    dx,
    dz,
    stabilizer_string,
):

    samples = dx.shape[0]

    sx, sz = prepare_zero_ancilla(
        samples
    )


    for q, component in enumerate(
        stabilizer_string
    ):

        if component == "I":
            continue


        if component == "X":

            noisy_h_data(
                dx,
                dz,
                q,
            )


        sx, sz = (
            noisy_data_to_syndrome_cnot(
                dx,
                dz,
                q,
                sx,
                sz,
            )
        )


        if component == "X":

            noisy_h_data(
                dx,
                dz,
                q,
            )


    # Z measurement is flipped by syndrome X/Y,
    # represented by sx=1.

    measurement = sx.clone()


    measurement = torch.logical_xor(
        measurement,
        bernoulli(
            measurement.shape,
            P_MEAS,
        ),
    )


    return (
        dx,
        dz,
        measurement,
    )


# ============================================================
# FLAGGED STABILIZER MEASUREMENT
#
# support interaction order:
#
#   d0
#   flag -> syndrome
#   d1
#   d2
#   flag -> syndrome
#   d3
#
# where d0..d3 are the four non-identity data positions
# of the stabilizer, in increasing physical-qubit order.
# ============================================================

def measure_stabilizer_flagged(
    dx,
    dz,
    stabilizer_string,
):

    samples = dx.shape[0]

    sx, sz = prepare_zero_ancilla(
        samples
    )

    fx, fz = prepare_plus_ancilla(
        samples
    )


    support = [
        q
        for q, component in enumerate(
            stabilizer_string
        )
        if component != "I"
    ]


    for interaction_index, q in enumerate(
        support
    ):

        component = (
            stabilizer_string[q]
        )


        if component == "X":

            noisy_h_data(
                dx,
                dz,
                q,
            )


        sx, sz = (
            noisy_data_to_syndrome_cnot(
                dx,
                dz,
                q,
                sx,
                sz,
            )
        )


        if component == "X":

            noisy_h_data(
                dx,
                dz,
                q,
            )


        if interaction_index == 0:

            (
                fx,
                fz,
                sx,
                sz,
            ) = (
                noisy_flag_to_syndrome_cnot(
                    fx,
                    fz,
                    sx,
                    sz,
                )
            )


        if interaction_index == 2:

            (
                fx,
                fz,
                sx,
                sz,
            ) = (
                noisy_flag_to_syndrome_cnot(
                    fx,
                    fz,
                    sx,
                    sz,
                )
            )


    # Syndrome ancilla measured in Z.
    syndrome_measurement = sx.clone()

    syndrome_measurement = (
        torch.logical_xor(
            syndrome_measurement,
            bernoulli(
                syndrome_measurement.shape,
                P_MEAS,
            ),
        )
    )


    # Flag ancilla measured in X.
    # Z/Y flips the X-basis result, represented by fz=1.
    flag_measurement = fz.clone()

    flag_measurement = (
        torch.logical_xor(
            flag_measurement,
            bernoulli(
                flag_measurement.shape,
                P_MEAS,
            ),
        )
    )


    return (
        dx,
        dz,
        syndrome_measurement,
        flag_measurement,
    )


# ============================================================
# EXTRACT ALL FOUR UNFLAGGED SYNDROMES
# ============================================================

def extract_all_unflagged(
    dx,
    dz,
):

    measured_bits = []


    for stabilizer_string in (
        STABILIZER_STRINGS
    ):

        (
            dx,
            dz,
            measured,
        ) = (
            measure_stabilizer_unflagged(
                dx,
                dz,
                stabilizer_string,
            )
        )


        measured_bits.append(
            measured
        )


    measured_bits = torch.stack(
        measured_bits,
        dim=1,
    )


    return (
        dx,
        dz,
        measured_bits,
    )


# ============================================================
# SYNDROME UTILITIES
# ============================================================

def true_syndrome_torch(
    dx,
    dz,
):

    batch_size = dx.shape[0]


    syndrome_bits = torch.zeros(
        batch_size,
        4,
        dtype=torch.bool,
        device=device,
    )


    for q in range(5):

        syndrome_bits = (
            torch.logical_xor(
                syndrome_bits,
                torch.logical_and(
                    dx[
                        :,
                        q,
                    ].unsqueeze(
                        1
                    ),
                    STAB_Z[
                        :,
                        q,
                    ].unsqueeze(
                        0
                    ),
                ),
            )
        )


        syndrome_bits = (
            torch.logical_xor(
                syndrome_bits,
                torch.logical_and(
                    dz[
                        :,
                        q,
                    ].unsqueeze(
                        1
                    ),
                    STAB_X[
                        :,
                        q,
                    ].unsqueeze(
                        0
                    ),
                ),
            )
        )


    return syndrome_bits


def syndrome_bits_to_index_torch(
    syndrome_bits,
):

    s = syndrome_bits.to(
        torch.long
    )


    return (
        8 * s[
            :,
            0,
        ]
        +
        4 * s[
            :,
            1,
        ]
        +
        2 * s[
            :,
            2,
        ]
        +
        s[
            :,
            3,
        ]
    )


# ============================================================
# APPLY A PAULI RECOVERY
# ============================================================

def apply_recovery(
    dx,
    dz,
    correction_x,
    correction_z,
):

    dx = torch.logical_xor(
        dx,
        correction_x,
    )

    dz = torch.logical_xor(
        dz,
        correction_z,
    )


    if USE_PAULI_FRAME_RECOVERY:

        return dx, dz


    recovery_mask = torch.logical_or(
        correction_x,
        correction_z,
    )


    fx, fz = (
        sample_single_qubit_depolarizing(
            dx.shape,
            P_RECOVERY,
        )
    )


    fx = torch.logical_and(
        fx,
        recovery_mask,
    )

    fz = torch.logical_and(
        fz,
        recovery_mask,
    )


    dx = torch.logical_xor(
        dx,
        fx,
    )

    dz = torch.logical_xor(
        dz,
        fz,
    )


    return dx, dz


# ============================================================
# CHARACTERIZE RESIDUAL PHYSICAL FRAME AS LOGICAL I/X/Y/Z
#
# We compose the noisy gadget output with an IDEAL final
# decoder only for channel characterization.
#
# A remaining correctable weight-one physical error therefore
# does NOT count as a logical failure.
# ============================================================

def residual_frame_to_logical_class(
    dx,
    dz,
):

    true_syn = true_syndrome_torch(
        dx,
        dz,
    )


    syn_index = (
        syndrome_bits_to_index_torch(
            true_syn
        )
    )


    canonical_x = DECODER_X[
        syn_index
    ]

    canonical_z = DECODER_Z[
        syn_index
    ]


    residual_x = torch.logical_xor(
        dx,
        canonical_x,
    )

    residual_z = torch.logical_xor(
        dz,
        canonical_z,
    )


    logical_x_bit = torch.zeros(
        dx.shape[0],
        dtype=torch.bool,
        device=device,
    )

    logical_z_bit = torch.zeros(
        dx.shape[0],
        dtype=torch.bool,
        device=device,
    )


    for q in range(5):

        logical_x_bit = torch.logical_xor(
            logical_x_bit,
            torch.logical_and(
                residual_x[
                    :,
                    q,
                ],
                LOGICAL_Z_Z[q],
            ),
        )

        logical_x_bit = torch.logical_xor(
            logical_x_bit,
            torch.logical_and(
                residual_z[
                    :,
                    q,
                ],
                LOGICAL_Z_X[q],
            ),
        )


        logical_z_bit = torch.logical_xor(
            logical_z_bit,
            torch.logical_and(
                residual_x[
                    :,
                    q,
                ],
                LOGICAL_X_Z[q],
            ),
        )

        logical_z_bit = torch.logical_xor(
            logical_z_bit,
            torch.logical_and(
                residual_z[
                    :,
                    q,
                ],
                LOGICAL_X_X[q],
            ),
        )


    logical_class = torch.zeros(
        dx.shape[0],
        dtype=torch.long,
        device=device,
    )


    logical_class[
        torch.logical_and(
            logical_x_bit,
            torch.logical_not(
                logical_z_bit
            ),
        )
    ] = 1


    logical_class[
        torch.logical_and(
            logical_x_bit,
            logical_z_bit,
        )
    ] = 2


    logical_class[
        torch.logical_and(
            torch.logical_not(
                logical_x_bit
            ),
            logical_z_bit,
        )
    ] = 3


    return logical_class


# ============================================================
# INITIAL PHYSICAL DATA NOISE
# ============================================================

def sample_initial_data_frame(
    samples,
    p_data,
):

    return (
        sample_single_qubit_depolarizing(
            (
                samples,
                5,
            ),
            p_data,
        )
    )


# ============================================================
# NAIVE ONE-ROUND NOISY QEC
#
# This is the direct baseline:
#
#   measure all four stabilizers once
#   with one syndrome ancilla each
#   use the ordinary minimum-weight decoder
# ============================================================

def naive_noisy_qec_distribution(
    p_data,
    samples,
    random_seed,
):

    torch.manual_seed(
        random_seed
    )

    torch.cuda.manual_seed_all(
        random_seed
    )


    dx, dz = (
        sample_initial_data_frame(
            samples,
            p_data,
        )
    )


    (
        dx,
        dz,
        measured_syndrome,
    ) = extract_all_unflagged(
        dx,
        dz,
    )


    syn_index = (
        syndrome_bits_to_index_torch(
            measured_syndrome
        )
    )


    correction_x = DECODER_X[
        syn_index
    ]

    correction_z = DECODER_Z[
        syn_index
    ]


    dx, dz = apply_recovery(
        dx,
        dz,
        correction_x,
        correction_z,
    )


    logical_class = (
        residual_frame_to_logical_class(
            dx,
            dz,
        )
    )


    counts = torch.bincount(
        logical_class,
        minlength=4,
    ).to(
        torch.float64
    )


    distribution = (
        counts
        /
        counts.sum()
    )


    return (
        distribution
        .cpu()
        .numpy()
    )


# ============================================================
# FLAGGED FTEC
#
# Published control logic:
#
# Measure stabilizers sequentially with a flag.
#
# If:
#   flag == 1
# OR
#   measured stabilizer syndrome == 1
#
# then:
#   extract all four syndromes using unflagged circuits.
#
# If flag == 1:
#   use the flag-aware lookup for the stabilizer that raised
#   the flag.
#
# Else:
#   use the ordinary minimum-weight decoder.
#
# After correction, that EC cycle ends.
#
# If all four flagged measurements are trivial and no flag
# fires, no correction is applied.
# ============================================================

def flagged_ftec_distribution(
    p_data,
    samples,
    random_seed,
):

    torch.manual_seed(
        random_seed
    )

    torch.cuda.manual_seed_all(
        random_seed
    )


    dx, dz = (
        sample_initial_data_frame(
            samples,
            p_data,
        )
    )


    active = torch.ones(
        samples,
        dtype=torch.bool,
        device=device,
    )


    flag_trigger_count = 0
    syndrome_trigger_count = 0


    for stabilizer_index, stabilizer_string in enumerate(
        STABILIZER_STRINGS
    ):

        active_indices = torch.where(
            active
        )[0]


        if active_indices.numel() == 0:
            break


        sub_dx = dx[
            active_indices
        ].clone()

        sub_dz = dz[
            active_indices
        ].clone()


        (
            sub_dx,
            sub_dz,
            measured_stabilizer,
            measured_flag,
        ) = measure_stabilizer_flagged(
            sub_dx,
            sub_dz,
            stabilizer_string,
        )


        dx[
            active_indices
        ] = sub_dx

        dz[
            active_indices
        ] = sub_dz


        trigger = torch.logical_or(
            measured_flag,
            measured_stabilizer,
        )


        if not torch.any(
            trigger
        ):
            continue


        local_trigger_indices = torch.where(
            trigger
        )[0]


        global_trigger_indices = (
            active_indices[
                local_trigger_indices
            ]
        )


        flag_for_triggered = (
            measured_flag[
                local_trigger_indices
            ]
        )


        flag_trigger_count += int(
            flag_for_triggered
            .sum()
            .item()
        )


        syndrome_trigger_count += int(
            torch.logical_and(
                torch.logical_not(
                    flag_for_triggered
                ),
                measured_stabilizer[
                    local_trigger_indices
                ],
            )
            .sum()
            .item()
        )


        trigger_dx = dx[
            global_trigger_indices
        ].clone()

        trigger_dz = dz[
            global_trigger_indices
        ].clone()


        # ----------------------------------------------------
        # Follow-up extraction of all four syndromes.
        # These circuits are ALSO noisy in this stochastic
        # experiment.
        #
        # In the one-fault proof, if the first flagged circuit
        # already used the one allowed fault, the follow-up
        # circuits are necessarily fault free. Here we allow
        # multiple stochastic faults naturally.
        # ----------------------------------------------------

        (
            trigger_dx,
            trigger_dz,
            full_syndrome,
        ) = extract_all_unflagged(
            trigger_dx,
            trigger_dz,
        )


        full_syn_index = (
            syndrome_bits_to_index_torch(
                full_syndrome
            )
        )


        standard_x = DECODER_X[
            full_syn_index
        ]

        standard_z = DECODER_Z[
            full_syn_index
        ]


        flagged_x = FLAG_CORR_X[
            stabilizer_index,
            full_syn_index,
        ]

        flagged_z = FLAG_CORR_Z[
            stabilizer_index,
            full_syn_index,
        ]


        correction_x = torch.where(
            flag_for_triggered.unsqueeze(
                1
            ),
            flagged_x,
            standard_x,
        )


        correction_z = torch.where(
            flag_for_triggered.unsqueeze(
                1
            ),
            flagged_z,
            standard_z,
        )


        trigger_dx, trigger_dz = (
            apply_recovery(
                trigger_dx,
                trigger_dz,
                correction_x,
                correction_z,
            )
        )


        dx[
            global_trigger_indices
        ] = trigger_dx

        dz[
            global_trigger_indices
        ] = trigger_dz


        # EC cycle ends for these samples.
        active[
            global_trigger_indices
        ] = False


    logical_class = (
        residual_frame_to_logical_class(
            dx,
            dz,
        )
    )


    counts = torch.bincount(
        logical_class,
        minlength=4,
    ).to(
        torch.float64
    )


    distribution = (
        counts
        /
        counts.sum()
    )


    diagnostics = {
        "flag_trigger_rate":
            flag_trigger_count
            /
            samples,

        "syndrome_trigger_rate":
            syndrome_trigger_count
            /
            samples,

        "no_trigger_rate":
            float(
                active
                .sum()
                .item()
            )
            /
            samples,
    }


    return (
        distribution
        .cpu()
        .numpy(),
        diagnostics,
    )


# ============================================================
# ESTIMATE LOGICAL CHANNELS
# ============================================================

print("\n======================================")
print("LOGICAL CHANNEL ESTIMATION")
print("======================================")

print(
    "BLOCK_SAMPLES:",
    BLOCK_SAMPLES,
)

print(
    "P_2Q:",
    P_2Q,
)

print(
    "P_1Q:",
    P_1Q,
)

print(
    "P_PREP:",
    P_PREP,
)

print(
    "P_MEAS:",
    P_MEAS,
)

print(
    "P_RECOVERY:",
    P_RECOVERY,
)

print(
    "USE_PAULI_FRAME_RECOVERY:",
    USE_PAULI_FRAME_RECOVERY,
)


LOGICAL_CHANNELS = {}


for p_index, p in enumerate(
    P_VALUES
):

    ideal_distribution = (
        ideal_five_qubit_logical_distribution(
            p
        )
    )


    naive_distribution = (
        naive_noisy_qec_distribution(
            p_data=p,
            samples=BLOCK_SAMPLES,
            random_seed=(
                SEED
                +
                10000 * p_index
                +
                100
            ),
        )
    )


    (
        flagged_distribution,
        flagged_diagnostics,
    ) = (
        flagged_ftec_distribution(
            p_data=p,
            samples=BLOCK_SAMPLES,
            random_seed=(
                SEED
                +
                10000 * p_index
                +
                500
            ),
        )
    )


    LOGICAL_CHANNELS[p] = {
        "ideal_code":
            ideal_distribution,

        "naive_qec":
            naive_distribution,

        "flagged_ftec":
            flagged_distribution,

        "diagnostics":
            flagged_diagnostics,
    }


    print(
        "\n--------------------------------------"
    )

    print(
        f"Physical p = {p:.4f}"
    )


    print(
        "Ideal code     | "
        f"I={ideal_distribution[0]:.6f} "
        f"X={ideal_distribution[1]:.6f} "
        f"Y={ideal_distribution[2]:.6f} "
        f"Z={ideal_distribution[3]:.6f} "
        f"pL={1.0 - ideal_distribution[0]:.6f}"
    )


    print(
        "Naive noisy EC | "
        f"I={naive_distribution[0]:.6f} "
        f"X={naive_distribution[1]:.6f} "
        f"Y={naive_distribution[2]:.6f} "
        f"Z={naive_distribution[3]:.6f} "
        f"pL={1.0 - naive_distribution[0]:.6f}"
    )


    print(
        "Flagged FTEC   | "
        f"I={flagged_distribution[0]:.6f} "
        f"X={flagged_distribution[1]:.6f} "
        f"Y={flagged_distribution[2]:.6f} "
        f"Z={flagged_distribution[3]:.6f} "
        f"pL={1.0 - flagged_distribution[0]:.6f}"
    )


    print(
        "Flag diagnostics | "
        f"flag="
        f"{flagged_diagnostics['flag_trigger_rate']:.4f} "
        f"syndrome="
        f"{flagged_diagnostics['syndrome_trigger_rate']:.4f} "
        f"no-trigger="
        f"{flagged_diagnostics['no_trigger_rate']:.4f}"
    )


# ============================================================
# FAST 8-LOGICAL-QUBIT STATEVECTOR
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

    batch_size = state.shape[0]

    left = (
        2 ** wire
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
        theta / 2
    )

    s = torch.sin(
        theta / 2
    )


    return torch.stack(
        [
            c * a
            -
            s * b,

            s * a
            +
            c * b,
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

    batch_size = state.shape[0]

    left = (
        2 ** wire
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

    batch_size = state.shape[0]

    left = (
        2 ** wire
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

    batch_size = state.shape[0]

    left = (
        2 ** wire
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
        & 1
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
        & 1
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
# LOGICAL PAULI SAMPLING
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
        cumulative[0]
    ] = 1


    codes[
        random_values
        >=
        cumulative[1]
    ] = 2


    codes[
        random_values
        >=
        cumulative[2]
    ] = 3


    return codes


def logical_distribution_for_mode(
    p,
    mode,
):

    if mode == "unprotected":

        return np.array(
            [
                1.0 - p,
                p / 3.0,
                p / 3.0,
                p / 3.0,
            ],
            dtype=np.float64,
        )


    if mode == "ideal_code":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "ideal_code"
            ]
        )


    if mode == "naive_qec":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "naive_qec"
            ]
        )


    if mode == "flagged_ftec":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "flagged_ftec"
            ]
        )


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# LOGICAL VQC
# ============================================================

def run_logical_quantum_circuit(
    inputs,
    quantum_weights,
    p,
    mode,
):

    batch_size = inputs.shape[0]

    state = initial_state(
        batch_size
    )


    distribution = (
        logical_distribution_for_mode(
            p,
            mode,
        )
    )


    encoding_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            distribution,
        )
    )


    ry_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            distribution,
        )
    )


    rz_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            distribution,
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
            distribution,
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


        for gate_index in range(
            N_QUBITS
        ):

            control = gate_index

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
# CLASSIFIER EVALUATION
# ============================================================

def evaluate_classifier(
    p,
    mode,
    random_seed,
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


            batch_size = x.shape[0]


            if (
                p == 0
                and
                mode in (
                    "unprotected",
                    "ideal_code",
                )
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
                    *
                    trajectories,
                    32,
                )
            )


            quantum_features = (
                run_logical_quantum_circuit(
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
# IDEAL CLASSIFIER BASELINE
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
# MAIN CIFAR-10 EXPERIMENT
# ============================================================

MODES = [
    "unprotected",
    "ideal_code",
    "naive_qec",
    "flagged_ftec",
]


print("\n======================================")
print("FLAGGED FTEC CIFAR-10 EXPERIMENT")
print("======================================")

print(
    "Trajectories:",
    N_TRAJECTORIES,
)

print(
    "Seeds:",
    N_SEEDS,
)


detailed_results = []


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
            1000 * p_index
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
                mode in (
                    "unprotected",
                    "ideal_code",
                )
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
                        100000
                        *
                        mode_index
                    ),
                )


            mode_results[
                mode
            ] = (
                accuracy,
                macro_f1,
                elapsed,
            )


        ideal_dist = (
            LOGICAL_CHANNELS[
                p
            ][
                "ideal_code"
            ]
        )

        naive_dist = (
            LOGICAL_CHANNELS[
                p
            ][
                "naive_qec"
            ]
        )

        flag_dist = (
            LOGICAL_CHANNELS[
                p
            ][
                "flagged_ftec"
            ]
        )

        diag = (
            LOGICAL_CHANNELS[
                p
            ][
                "diagnostics"
            ]
        )


        detailed_results.append(
            {
                "p":
                    p,

                "seed":
                    base_seed,

                "ideal_code_logical_error":
                    1.0
                    -
                    ideal_dist[0],

                "naive_qec_logical_error":
                    1.0
                    -
                    naive_dist[0],

                "flagged_ftec_logical_error":
                    1.0
                    -
                    flag_dist[0],

                "flagged_logical_X":
                    flag_dist[1],

                "flagged_logical_Y":
                    flag_dist[2],

                "flagged_logical_Z":
                    flag_dist[3],

                "flag_trigger_rate":
                    diag[
                        "flag_trigger_rate"
                    ],

                "syndrome_trigger_rate":
                    diag[
                        "syndrome_trigger_rate"
                    ],

                "no_trigger_rate":
                    diag[
                        "no_trigger_rate"
                    ],

                "unprotected_accuracy":
                    mode_results[
                        "unprotected"
                    ][0],

                "ideal_code_accuracy":
                    mode_results[
                        "ideal_code"
                    ][0],

                "naive_qec_accuracy":
                    mode_results[
                        "naive_qec"
                    ][0],

                "flagged_ftec_accuracy":
                    mode_results[
                        "flagged_ftec"
                    ][0],

                "unprotected_f1":
                    mode_results[
                        "unprotected"
                    ][1],

                "ideal_code_f1":
                    mode_results[
                        "ideal_code"
                    ][1],

                "naive_qec_f1":
                    mode_results[
                        "naive_qec"
                    ][1],

                "flagged_ftec_f1":
                    mode_results[
                        "flagged_ftec"
                    ][1],
            }
        )


        print(
            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "

            f"None="
            f"{mode_results['unprotected'][0]:.4f} | "

            f"Ideal5="
            f"{mode_results['ideal_code'][0]:.4f} | "

            f"Naive="
            f"{mode_results['naive_qec'][0]:.4f} | "

            f"FlagFT="
            f"{mode_results['flagged_ftec'][0]:.4f}"
        )


# ============================================================
# SAVE DETAILED CSV
# ============================================================

DETAILED_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_flagged_5qubit_ftec_detailed.csv",
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

        "ideal_code_logical_error":
            subset[0][
                "ideal_code_logical_error"
            ],

        "naive_qec_logical_error":
            subset[0][
                "naive_qec_logical_error"
            ],

        "flagged_ftec_logical_error":
            subset[0][
                "flagged_ftec_logical_error"
            ],

        "flag_trigger_rate":
            subset[0][
                "flag_trigger_rate"
            ],

        "syndrome_trigger_rate":
            subset[0][
                "syndrome_trigger_rate"
            ],

        "no_trigger_rate":
            subset[0][
                "no_trigger_rate"
            ],
    }


    for key in [
        "unprotected_accuracy",
        "ideal_code_accuracy",
        "naive_qec_accuracy",
        "flagged_ftec_accuracy",
    ]:

        values = np.array(
            [
                row[key]
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
    "cifar10_flagged_5qubit_ftec_summary.csv",
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
# PRINT FINAL SUMMARY
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Unprotected\t"
    "Ideal [[5,1,3]]\t"
    "Naive noisy QEC\t"
    "Flagged FTEC"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['unprotected_accuracy_mean']:.4f}"
        f" ± "
        f"{row['unprotected_accuracy_std']:.4f}\t"

        f"{row['ideal_code_accuracy_mean']:.4f}"
        f" ± "
        f"{row['ideal_code_accuracy_std']:.4f}\t"

        f"{row['naive_qec_accuracy_mean']:.4f}"
        f" ± "
        f"{row['naive_qec_accuracy_std']:.4f}\t"

        f"{row['flagged_ftec_accuracy_mean']:.4f}"
        f" ± "
        f"{row['flagged_ftec_accuracy_std']:.4f}"
    )


print("\n======================================")
print("LOGICAL ERROR SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Ideal pL\t"
    "Naive pL\t"
    "Flagged pL"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['ideal_code_logical_error']:.6f}\t"

        f"{row['naive_qec_logical_error']:.6f}\t"

        f"{row['flagged_ftec_logical_error']:.6f}"
    )


print("\n======================================")
print("FLAG / BRANCH SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Flag trigger\t"
    "Syndrome trigger\t"
    "No trigger"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['flag_trigger_rate']:.4f}\t\t"

        f"{row['syndrome_trigger_rate']:.4f}\t\t"

        f"{row['no_trigger_rate']:.4f}"
    )


# ============================================================
# PLOT ARRAYS
# ============================================================

p_values = np.array(
    [
        row["p"]
        for row in summary
    ]
)


def summary_array(
    key,
):

    return np.array(
        [
            row[key]
            for row in summary
        ]
    )


none_mean = summary_array(
    "unprotected_accuracy_mean"
)

none_std = summary_array(
    "unprotected_accuracy_std"
)

ideal_mean = summary_array(
    "ideal_code_accuracy_mean"
)

ideal_std = summary_array(
    "ideal_code_accuracy_std"
)

naive_mean = summary_array(
    "naive_qec_accuracy_mean"
)

naive_std = summary_array(
    "naive_qec_accuracy_std"
)

flag_mean = summary_array(
    "flagged_ftec_accuracy_mean"
)

flag_std = summary_array(
    "flagged_ftec_accuracy_std"
)


ideal_pl = summary_array(
    "ideal_code_logical_error"
)

naive_pl = summary_array(
    "naive_qec_logical_error"
)

flag_pl = summary_array(
    "flagged_ftec_logical_error"
)


# ============================================================
# FIGURE 1:
# ACCURACY COMPARISON
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


plt.plot(
    p_values
    *
    100,
    none_mean
    *
    100,
    marker="o",
    linewidth=2,
    label="Unprotected",
)


plt.plot(
    p_values
    *
    100,
    ideal_mean
    *
    100,
    marker="s",
    linewidth=2,
    label="Ideal [[5,1,3]] QEC",
)


plt.plot(
    p_values
    *
    100,
    naive_mean
    *
    100,
    marker="^",
    linewidth=2,
    label="Naive noisy stabilizer QEC",
)


plt.plot(
    p_values
    *
    100,
    flag_mean
    *
    100,
    marker="D",
    linewidth=2,
    label="Flagged [[5,1,3]] FTEC",
)


for mean, std in [
    (
        none_mean,
        none_std,
    ),
    (
        ideal_mean,
        ideal_std,
    ),
    (
        naive_mean,
        naive_std,
    ),
    (
        flag_mean,
        flag_std,
    ),
]:

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
    "Flagged [[5,1,3]] Fault-Tolerant Error Correction"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_flagged_5qubit_ftec_accuracy.png",
)


plt.savefig(
    ACCURACY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2:
# LOGICAL ERROR
# ============================================================

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
    p_values
    *
    100,
    marker="o",
    linewidth=2,
    label="Unprotected physical p",
)


plt.plot(
    p_values
    *
    100,
    ideal_pl
    *
    100,
    marker="s",
    linewidth=2,
    label="Ideal [[5,1,3]]",
)


plt.plot(
    p_values
    *
    100,
    naive_pl
    *
    100,
    marker="^",
    linewidth=2,
    label="Naive noisy QEC",
)


plt.plot(
    p_values
    *
    100,
    flag_pl
    *
    100,
    marker="D",
    linewidth=2,
    label="Flagged FTEC",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Residual Logical Pauli Error Probability (%)"
)

plt.title(
    "Logical Error Suppression with Flagged FTEC"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


LOGICAL_ERROR_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_flagged_ftec_logical_error.png",
)


plt.savefig(
    LOGICAL_ERROR_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3:
# FLAGGED FTEC GAIN OVER NAIVE QEC
# ============================================================

gain = (
    flag_mean
    -
    naive_mean
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
    gain
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
    "Flagged FTEC − Naive QEC Accuracy "
    "(percentage points)"
)

plt.title(
    "Classification Benefit of Flagged Syndrome Extraction"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GAIN_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_flagged_ftec_gain.png",
)


plt.savefig(
    GAIN_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 4:
# FLAG / BRANCH RATES
# ============================================================

flag_rate = summary_array(
    "flag_trigger_rate"
)

syndrome_rate = summary_array(
    "syndrome_trigger_rate"
)

no_trigger_rate = summary_array(
    "no_trigger_rate"
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
    flag_rate
    *
    100,
    marker="o",
    linewidth=2,
    label="Flag branch",
)


plt.plot(
    p_values
    *
    100,
    syndrome_rate
    *
    100,
    marker="s",
    linewidth=2,
    label="Nontrivial-syndrome branch",
)


plt.plot(
    p_values
    *
    100,
    no_trigger_rate
    *
    100,
    marker="^",
    linewidth=2,
    label="No branch",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Fraction of QEC Cycles (%)"
)

plt.title(
    "Adaptive Branching in Flagged [[5,1,3]] FTEC"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


BRANCH_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_flagged_ftec_branch_rates.png",
)


plt.savefig(
    BRANCH_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 5:
# FLAGGED LOGICAL PAULI DISTRIBUTION
# ============================================================

flagged_x = np.array(
    [
        LOGICAL_CHANNELS[p][
            "flagged_ftec"
        ][1]
        for p in P_VALUES
    ]
)

flagged_y = np.array(
    [
        LOGICAL_CHANNELS[p][
            "flagged_ftec"
        ][2]
        for p in P_VALUES
    ]
)

flagged_z = np.array(
    [
        LOGICAL_CHANNELS[p][
            "flagged_ftec"
        ][3]
        for p in P_VALUES
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
    flagged_x
    *
    100,
    marker="o",
    linewidth=2,
    label="Logical X",
)


plt.plot(
    p_values
    *
    100,
    flagged_y
    *
    100,
    marker="s",
    linewidth=2,
    label="Logical Y",
)


plt.plot(
    p_values
    *
    100,
    flagged_z
    *
    100,
    marker="^",
    linewidth=2,
    label="Logical Z",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Residual Logical Pauli Probability (%)"
)

plt.title(
    "Residual Logical Channel After Flagged FTEC"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


PAULI_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_flagged_ftec_pauli_distribution.png",
)


plt.savefig(
    PAULI_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FILE SUMMARY
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
    "Accuracy figure:",
    ACCURACY_FIGURE,
)

print(
    "Logical-error figure:",
    LOGICAL_ERROR_FIGURE,
)

print(
    "Flagged gain figure:",
    GAIN_FIGURE,
)

print(
    "Branch-rate figure:",
    BRANCH_FIGURE,
)

print(
    "Logical-Pauli figure:",
    PAULI_FIGURE,
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "The naive baseline measures all four stabilizers "
    "using one syndrome ancilla and has no flag qubit."
)

print(
    "The flagged FTEC circuit uses one syndrome ancilla "
    "and one flag ancilla for each sequential stabilizer "
    "measurement."
)

print(
    "If the flag fires or a stabilizer result is "
    "nontrivial, all four syndromes are extracted and "
    "the decoder branches accordingly."
)

print(
    "Flagged correlated single-fault errors are decoded "
    "with an automatically generated flag-aware lookup "
    "table."
)

print(
    "The XZZXI lookup is checked against the expected "
    "8-class flagged correction set."
)

print(
    "Follow-up syndrome circuits remain noisy, so the "
    "Monte-Carlo experiment allows multiple physical "
    "faults rather than assuming the one-fault proof case."
)

print(
    "The CIFAR classifier remains an eight-logical-qubit "
    "statevector model; the five-qubit physical blocks "
    "are represented through their estimated logical "
    "Pauli channels."
)

print(
    "This upgrades the error-correction gadget to a "
    "flag-aware distance-three FTEC model, but arbitrary "
    "logical RY/RZ gates are still abstracted."
)
