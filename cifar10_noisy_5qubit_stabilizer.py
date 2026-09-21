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

# Number of encoded-block samples used to estimate the
# circuit-level logical Pauli channel.
BLOCK_SAMPLES = 500_000


# ============================================================
# PHYSICAL DATA-NOISE SWEEP
#
# Each physical data qubit in one [[5,1,3]] block receives
# depolarizing noise with probability p.
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
# NOISY STABILIZER-EXTRACTION SETTINGS
#
# P_2Q:
# Probability that a syndrome-extraction CNOT suffers a
# two-qubit depolarizing fault. Conditional on a fault,
# one of the 15 non-II two-qubit Paulis is chosen uniformly.
#
# P_1Q:
# Depolarizing error after each basis-changing H used for
# measuring X components of a stabilizer.
#
# P_PREP:
# Depolarizing error on a freshly prepared syndrome ancilla.
#
# P_MEAS:
# Classical bit-flip probability on ancilla measurement.
#
# P_RECOVERY:
# Depolarizing error after an executed physical Pauli recovery.
# ============================================================

P_2Q = 0.005
P_1Q = 0.001
P_PREP = 0.001
P_MEAS = 0.010
P_RECOVERY = 0.005


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
    raise RuntimeError("CUDA is required for this experiment.")

device = torch.device("cuda")


print("\n======================================")
print("DEVICE")
print("======================================")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)


# ============================================================
# DATA
# ============================================================

data = np.load(DATA_PATH, allow_pickle=True)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"].astype(np.int64)

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
            0.05 * torch.randn(
                N_UPLOADS,
                N_QUBITS,
                2,
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, N_CLASSES),
        )


model = QuantumClassifier()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

model.load_state_dict(checkpoint)
model = model.to(device)

for parameter in model.parameters():
    parameter.requires_grad = False

model.eval()


print("\n======================================")
print("CHECKPOINT")
print("======================================")
print("Loaded:", MODEL_PATH)
print("Quantum weights:", model.quantum_weights.device)
print("Classifier:", model.classifier[0].weight.device)


# ============================================================
# [[5,1,3]] PERFECT CODE
#
# Stabilizer generators:
#
# g1 = X Z Z X I
# g2 = I X Z Z X
# g3 = X I X Z Z
# g4 = Z X I X Z
#
# Logical X = X X X X X
# Logical Z = Z Z Z Z Z
#
# Pauli code convention:
#
# 0 = I
# 1 = X
# 2 = Y
# 3 = Z
# ============================================================

STABILIZER_STRINGS = [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ",
]

LOGICAL_X_STRING = "XXXXX"
LOGICAL_Z_STRING = "ZZZZZ"

PAULI_CHARS = ["I", "X", "Y", "Z"]


def pauli_string_to_symplectic(pauli_string):

    x = np.zeros(
        len(pauli_string),
        dtype=np.uint8,
    )

    z = np.zeros(
        len(pauli_string),
        dtype=np.uint8,
    )

    for i, p in enumerate(pauli_string):

        if p == "X":
            x[i] = 1

        elif p == "Y":
            x[i] = 1
            z[i] = 1

        elif p == "Z":
            z[i] = 1

    return x, z


STABILIZERS = [
    pauli_string_to_symplectic(s)
    for s in STABILIZER_STRINGS
]

LOGICAL_X = pauli_string_to_symplectic(
    LOGICAL_X_STRING
)

LOGICAL_Z = pauli_string_to_symplectic(
    LOGICAL_Z_STRING
)

ZERO_PAULI = (
    np.zeros(5, dtype=np.uint8),
    np.zeros(5, dtype=np.uint8),
)


def symplectic_inner(p1, p2):

    x1, z1 = p1
    x2, z2 = p2

    return int(
        (
            np.dot(x1, z2)
            +
            np.dot(z1, x2)
        )
        % 2
    )


def syndrome_numpy(pauli):

    return tuple(
        symplectic_inner(
            pauli,
            stabilizer,
        )
        for stabilizer in STABILIZERS
    )


def multiply_paulis(p1, p2):

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


# ============================================================
# MINIMUM-WEIGHT SYNDROME DECODER
# ============================================================

DECODER = {}

for qubit in range(5):

    for pauli_char in ["X", "Y", "Z"]:

        error = ["I"] * 5
        error[qubit] = pauli_char

        error = pauli_string_to_symplectic(
            "".join(error)
        )

        DECODER[
            syndrome_numpy(error)
        ] = error

assert len(DECODER) == 15


# ============================================================
# SYNDROME LOOKUP ARRAYS
#
# syndrome integer:
#
# s0 s1 s2 s3 -> binary integer 0..15
#
# index 0 = no correction
# indices 1..15 = unique single-qubit correction
# ============================================================

DECODER_X_NP = np.zeros(
    (16, 5),
    dtype=np.uint8,
)

DECODER_Z_NP = np.zeros(
    (16, 5),
    dtype=np.uint8,
)


def syndrome_tuple_to_index(s):

    return (
        8 * s[0]
        +
        4 * s[1]
        +
        2 * s[2]
        +
        s[3]
    )


for syn, correction in DECODER.items():

    idx = syndrome_tuple_to_index(syn)

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


# Keep symplectic masks as Boolean tensors.
#
# This deliberately avoids integer matrix multiplication on CUDA.
# Some PyTorch/CUDA combinations (including the one on this machine)
# do not implement addmm/matmul for torch.int32.
STAB_X = torch.tensor(
    np.stack(
        [s[0] for s in STABILIZERS]
    ),
    dtype=torch.bool,
    device=device,
)

STAB_Z = torch.tensor(
    np.stack(
        [s[1] for s in STABILIZERS]
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
# PHYSICAL PAULI PATTERN -> LOGICAL PAULI
#
# Used for exact ideal-code verification.
#
# output:
# 0 = I_L
# 1 = X_L
# 2 = Y_L
# 3 = Z_L
# ============================================================

def physical_pattern_to_logical(pattern):

    pauli_string = "".join(
        PAULI_CHARS[code]
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

    if syn == (0, 0, 0, 0):
        correction = ZERO_PAULI
    else:
        correction = DECODER[syn]

    residual = multiply_paulis(
        correction,
        physical_error,
    )

    assert syndrome_numpy(
        residual
    ) == (
        0,
        0,
        0,
        0,
    )

    logical_x_bit = (
        symplectic_inner(
            residual,
            LOGICAL_Z,
        )
    )

    logical_z_bit = (
        symplectic_inner(
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


# ============================================================
# EXACT IDEAL [[5,1,3]] LOGICAL CHANNEL
# ============================================================

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


def ideal_five_qubit_logical_distribution(p):

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
            ** (5 - weight)
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
# VERIFY THE PREVIOUS OUTPUT
# ============================================================

print("\n======================================")
print("IDEAL [[5,1,3]] VERIFICATION")
print("======================================")

EXPECTED_PREVIOUS_PL = {
    0.000: 0.000000,
    0.001: 0.000010,
    0.002: 0.000040,
    0.005: 0.000247,
    0.010: 0.000978,
    0.020: 0.003825,
    0.050: 0.022332,
}

for p in P_VALUES:

    dist = (
        ideal_five_qubit_logical_distribution(
            p
        )
    )

    p_logical = (
        1.0
        -
        dist[0]
    )

    expected = (
        EXPECTED_PREVIOUS_PL[p]
    )

    print(
        f"p={p:.4f} | "
        f"computed pL={p_logical:.8f} | "
        f"previous≈{expected:.6f} | "
        f"difference="
        f"{abs(p_logical - expected):.2e}"
    )


# ============================================================
# PAULI NOISE SAMPLERS FOR THE PHYSICAL CODE BLOCK
# ============================================================

def sample_single_qubit_depolarizing(
    shape,
    p,
):

    """
    Returns two Boolean tensors (x,z).

    I : x=0 z=0
    X : x=1 z=0
    Y : x=1 z=1
    Z : x=0 z=1
    """

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


    has_error = (
        torch.rand(
            shape,
            device=device,
        )
        < p
    )


    pauli_type = torch.randint(
        low=0,
        high=3,
        size=shape,
        device=device,
    )


    # type 0 -> X
    # type 1 -> Y
    # type 2 -> Z

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

    """
    Two-qubit depolarizing fault.

    With probability 1-p:
        II

    With probability p:
        uniformly sample one of the 15 non-II
        Pauli products.

    Returns:
        x1,z1,x2,z2
    """

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


    has_error = (
        torch.rand(
            shape,
            device=device,
        )
        < p
    )


    # 1..15 encodes all two-qubit Pauli pairs
    # except II:
    #
    # code = 4 * first_pauli + second_pauli
    #
    # Pauli:
    # 0 I
    # 1 X
    # 2 Y
    # 3 Z

    pair_code = torch.randint(
        low=1,
        high=16,
        size=shape,
        device=device,
    )


    p1 = (
        pair_code
        // 4
    )

    p2 = (
        pair_code
        % 4
    )


    x1 = torch.logical_and(
        has_error,
        torch.logical_or(
            p1 == 1,
            p1 == 2,
        ),
    )

    z1 = torch.logical_and(
        has_error,
        torch.logical_or(
            p1 == 2,
            p1 == 3,
        ),
    )


    x2 = torch.logical_and(
        has_error,
        torch.logical_or(
            p2 == 1,
            p2 == 2,
        ),
    )

    z2 = torch.logical_and(
        has_error,
        torch.logical_or(
            p2 == 2,
            p2 == 3,
        ),
    )


    return (
        x1,
        z1,
        x2,
        z2,
    )


# ============================================================
# PAULI-FRAME GATE PROPAGATION
# ============================================================

def noisy_h_on_data(
    dx,
    dz,
    q,
):

    """
    Ideal H:
        X <-> Z

    Then one-qubit depolarizing noise P_1Q.
    """

    old_x = dx[:, q].clone()

    dx[:, q] = dz[:, q]
    dz[:, q] = old_x


    fx, fz = (
        sample_single_qubit_depolarizing(
            dx[:, q].shape,
            P_1Q,
        )
    )

    dx[:, q] = torch.logical_xor(
        dx[:, q],
        fx,
    )

    dz[:, q] = torch.logical_xor(
        dz[:, q],
        fz,
    )


def noisy_data_to_ancilla_cnot(
    dx,
    dz,
    q,
    ax,
    az,
):

    """
    CNOT:
        data q = control
        ancilla = target

    Pauli propagation:

    X_control -> X_control X_target
    Z_target  -> Z_control Z_target

    Then a two-qubit depolarizing fault.
    """

    control_x = dx[:, q].clone()
    control_z = dz[:, q].clone()

    target_x = ax.clone()
    target_z = az.clone()


    dx[:, q] = control_x

    dz[:, q] = torch.logical_xor(
        control_z,
        target_z,
    )

    ax = torch.logical_xor(
        target_x,
        control_x,
    )

    az = target_z


    (
        fx_data,
        fz_data,
        fx_anc,
        fz_anc,
    ) = sample_two_qubit_depolarizing(
        ax.shape,
        P_2Q,
    )


    dx[:, q] = torch.logical_xor(
        dx[:, q],
        fx_data,
    )

    dz[:, q] = torch.logical_xor(
        dz[:, q],
        fz_data,
    )

    ax = torch.logical_xor(
        ax,
        fx_anc,
    )

    az = torch.logical_xor(
        az,
        fz_anc,
    )


    return ax, az


# ============================================================
# ONE NOISY STABILIZER MEASUREMENT
#
# For a Z component:
#     data -> ancilla CNOT
#
# For an X component:
#     H(data)
#     data -> ancilla CNOT
#     H(data)
#
# Fresh ancilla begins in |0>.
#
# The ancilla is then measured in Z.
# ============================================================

def measure_stabilizer_noisy(
    dx,
    dz,
    stabilizer_string,
):

    samples = dx.shape[0]


    # --------------------------------------------------------
    # Fresh ancilla |0> with preparation noise.
    # --------------------------------------------------------

    ax, az = (
        sample_single_qubit_depolarizing(
            (samples,),
            P_PREP,
        )
    )


    # --------------------------------------------------------
    # Weight-4 stabilizer interaction
    # --------------------------------------------------------

    for q, component in enumerate(
        stabilizer_string
    ):

        if component == "I":
            continue


        if component == "X":

            noisy_h_on_data(
                dx,
                dz,
                q,
            )


        # component is now effectively Z in the
        # measurement basis.

        ax, az = (
            noisy_data_to_ancilla_cnot(
                dx,
                dz,
                q,
                ax,
                az,
            )
        )


        if component == "X":

            noisy_h_on_data(
                dx,
                dz,
                q,
            )


    # --------------------------------------------------------
    # Z measurement outcome.
    #
    # Ancilla X or Y component flips a Z measurement,
    # represented by ax=1.
    # --------------------------------------------------------

    measurement = ax.clone()


    classical_flip = (
        torch.rand(
            measurement.shape,
            device=device,
        )
        < P_MEAS
    )


    measurement = torch.logical_xor(
        measurement,
        classical_flip,
    )


    return measurement


# ============================================================
# VECTORISED TRUE SYNDROME OF A PHYSICAL PAULI FRAME
# ============================================================

def true_syndrome_torch(
    dx,
    dz,
):

    # --------------------------------------------------------
    # Symplectic syndrome:
    #
    # s_j = x_error . z_stabilizer
    #       XOR
    #       z_error . x_stabilizer
    #
    # Everything is Boolean here. This is equivalent to the
    # mod-2 dot product but avoids CUDA int32 matrix multiply.
    #
    # Shapes:
    # dx, dz   : [batch, 5]
    # STAB_*   : [4, 5]
    # result   : [batch, 4]
    # --------------------------------------------------------

    batch_size = dx.shape[0]

    syndrome_bits = torch.zeros(
        batch_size,
        4,
        dtype=torch.bool,
        device=device,
    )

    for q in range(5):

        # Contribution x_error[q] * z_stabilizer[:, q]
        syndrome_bits = torch.logical_xor(
            syndrome_bits,
            torch.logical_and(
                dx[:, q].unsqueeze(1),
                STAB_Z[:, q].unsqueeze(0),
            ),
        )

        # Contribution z_error[q] * x_stabilizer[:, q]
        syndrome_bits = torch.logical_xor(
            syndrome_bits,
            torch.logical_and(
                dz[:, q].unsqueeze(1),
                STAB_X[:, q].unsqueeze(0),
            ),
        )

    return syndrome_bits


def syndrome_bits_to_index(
    syndrome_bits,
):

    s = syndrome_bits.to(
        torch.long
    )

    return (
        8 * s[:, 0]
        +
        4 * s[:, 1]
        +
        2 * s[:, 2]
        +
        s[:, 3]
    )


# ============================================================
# APPLY DECODER RECOVERY
# ============================================================

def apply_noisy_recovery(
    dx,
    dz,
    decoded_syndrome,
):

    decoder_index = (
        syndrome_bits_to_index(
            decoded_syndrome
        )
    )


    correction_x = (
        DECODER_X[
            decoder_index
        ]
    )

    correction_z = (
        DECODER_Z[
            decoder_index
        ]
    )


    # Ideal requested correction

    dx = torch.logical_xor(
        dx,
        correction_x,
    )

    dz = torch.logical_xor(
        dz,
        correction_z,
    )


    # --------------------------------------------------------
    # Noisy physical recovery operation.
    #
    # Only the qubit on which a non-identity correction
    # was requested receives recovery-gate noise.
    # --------------------------------------------------------

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
# MAP ANY RESIDUAL PHYSICAL PAULI FRAME TO A LOGICAL PAULI
#
# We characterize the noisy EC gadget by composing it with an
# IDEAL final decoder.
#
# This is important:
#
# A residual correctable physical error after noisy EC is not
# automatically counted as a logical failure.
#
# The ideal decoder first removes the residual syndrome.
# The remaining normalizer element is classified as
# I_L, X_L, Y_L, or Z_L.
# ============================================================

def residual_frame_to_logical_class(
    dx,
    dz,
):

    syn = true_syndrome_torch(
        dx,
        dz,
    )


    syn_index = (
        syndrome_bits_to_index(
            syn
        )
    )


    canonical_x = (
        DECODER_X[
            syn_index
        ]
    )

    canonical_z = (
        DECODER_Z[
            syn_index
        ]
    )


    residual_x = torch.logical_xor(
        dx,
        canonical_x,
    )

    residual_z = torch.logical_xor(
        dz,
        canonical_z,
    )


    # --------------------------------------------------------
    # Logical commutation tests, again using pure Boolean XOR
    # rather than integer reductions. This keeps the complete
    # stabilizer decoder CUDA-safe on this PyTorch build.
    # --------------------------------------------------------

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

        # Logical X component:
        # residual anticommutes with logical Z
        logical_x_bit = torch.logical_xor(
            logical_x_bit,
            torch.logical_and(
                residual_x[:, q],
                LOGICAL_Z_Z[q],
            ),
        )

        logical_x_bit = torch.logical_xor(
            logical_x_bit,
            torch.logical_and(
                residual_z[:, q],
                LOGICAL_Z_X[q],
            ),
        )

        # Logical Z component:
        # residual anticommutes with logical X
        logical_z_bit = torch.logical_xor(
            logical_z_bit,
            torch.logical_and(
                residual_x[:, q],
                LOGICAL_X_Z[q],
            ),
        )

        logical_z_bit = torch.logical_xor(
            logical_z_bit,
            torch.logical_and(
                residual_z[:, q],
                LOGICAL_X_X[q],
            ),
        )


    logical_class = torch.zeros(
        dx.shape[0],
        dtype=torch.long,
        device=device,
    )


    # X
    logical_class[
        torch.logical_and(
            logical_x_bit == 1,
            logical_z_bit == 0,
        )
    ] = 1


    # Y
    logical_class[
        torch.logical_and(
            logical_x_bit == 1,
            logical_z_bit == 1,
        )
    ] = 2


    # Z
    logical_class[
        torch.logical_and(
            logical_x_bit == 0,
            logical_z_bit == 1,
        )
    ] = 3


    return logical_class


# ============================================================
# CIRCUIT-LEVEL NOISY [[5,1,3]] EC CHANNEL
#
# 1) Five physical data qubits receive depolarizing noise p.
#
# 2) Four stabilizers are measured with explicit noisy
#    ancilla circuits.
#
# 3) For rounds=3, all four syndrome bits are measured three
#    times and majority voted independently.
#
# 4) A minimum-weight recovery is applied and is itself noisy.
#
# 5) An ideal final decoder identifies the residual logical
#    I/X/Y/Z channel.
# ============================================================

def noisy_five_qubit_logical_distribution(
    p_data,
    rounds,
    samples,
    random_seed,
):

    torch.manual_seed(
        random_seed
    )

    torch.cuda.manual_seed_all(
        random_seed
    )


    # --------------------------------------------------------
    # Initial physical data errors
    # --------------------------------------------------------

    dx, dz = (
        sample_single_qubit_depolarizing(
            (samples, 5),
            p_data,
        )
    )


    round_syndromes = []


    # --------------------------------------------------------
    # Repeated syndrome extraction
    #
    # Data errors persist between rounds.
    # Syndrome circuitry can therefore introduce new data
    # errors as the rounds proceed.
    # --------------------------------------------------------

    for _ in range(rounds):

        syndrome_measurements = []


        for stabilizer_string in STABILIZER_STRINGS:

            measured_bit = (
                measure_stabilizer_noisy(
                    dx,
                    dz,
                    stabilizer_string,
                )
            )

            syndrome_measurements.append(
                measured_bit
            )


        syndrome_measurements = (
            torch.stack(
                syndrome_measurements,
                dim=1,
            )
        )


        round_syndromes.append(
            syndrome_measurements
        )


    # [samples, rounds, 4]

    syndrome_history = torch.stack(
        round_syndromes,
        dim=1,
    )


    threshold = (
        rounds // 2
        + 1
    )


    decoded_syndrome = (

        syndrome_history
        .sum(dim=1)

        >= threshold
    )


    # --------------------------------------------------------
    # Noisy recovery
    # --------------------------------------------------------

    dx, dz = apply_noisy_recovery(
        dx,
        dz,
        decoded_syndrome,
    )


    # --------------------------------------------------------
    # Compose with ideal decoder and classify residual logical
    # channel.
    # --------------------------------------------------------

    logical_class = (
        residual_frame_to_logical_class(
            dx,
            dz,
        )
    )


    counts = torch.bincount(
        logical_class,
        minlength=4,
    ).to(torch.float64)


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
# ESTIMATE ALL LOGICAL CHANNELS
# ============================================================

print("\n======================================")
print("NOISY STABILIZER CHANNEL ESTIMATION")
print("======================================")

print("Block samples:", BLOCK_SAMPLES)
print("P_2Q:", P_2Q)
print("P_1Q:", P_1Q)
print("P_PREP:", P_PREP)
print("P_MEAS:", P_MEAS)
print("P_RECOVERY:", P_RECOVERY)


LOGICAL_CHANNELS = {}


for p_index, p in enumerate(P_VALUES):

    ideal_dist = (
        ideal_five_qubit_logical_distribution(
            p
        )
    )


    qec1_dist = (
        noisy_five_qubit_logical_distribution(
            p_data=p,
            rounds=1,
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


    qec3_dist = (
        noisy_five_qubit_logical_distribution(
            p_data=p,
            rounds=3,
            samples=BLOCK_SAMPLES,
            random_seed=(
                SEED
                +
                10000 * p_index
                +
                300
            ),
        )
    )


    LOGICAL_CHANNELS[p] = {
        "ideal_code": ideal_dist,
        "qec1": qec1_dist,
        "qec3": qec3_dist,
    }


    print(
        "\n"
        f"p={p:.4f}"
    )

    print(
        "Ideal code : "
        f"I={ideal_dist[0]:.6f} "
        f"X={ideal_dist[1]:.6f} "
        f"Y={ideal_dist[2]:.6f} "
        f"Z={ideal_dist[3]:.6f} "
        f"pL={1-ideal_dist[0]:.6f}"
    )

    print(
        "Noisy QEC-1: "
        f"I={qec1_dist[0]:.6f} "
        f"X={qec1_dist[1]:.6f} "
        f"Y={qec1_dist[2]:.6f} "
        f"Z={qec1_dist[3]:.6f} "
        f"pL={1-qec1_dist[0]:.6f}"
    )

    print(
        "Noisy QEC-3: "
        f"I={qec3_dist[0]:.6f} "
        f"X={qec3_dist[1]:.6f} "
        f"Y={qec3_dist[2]:.6f} "
        f"Z={qec3_dist[3]:.6f} "
        f"pL={1-qec3_dist[0]:.6f}"
    )


# ============================================================
# FAST 8-LOGICAL-QUBIT STATEVECTOR SIMULATOR
# ============================================================

def initial_state(batch_size):

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
        2 ** (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    x = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


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
        2 ** (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    x = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


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
        2 ** (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    x = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


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
        2 ** (
            N_QUBITS
            -
            wire
            -
            1
        )
    )


    x = state.reshape(
        batch_size,
        left,
        2,
        right,
    )


    a = x[:, :, 0, :]
    b = x[:, :, 1, :]


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
# CNOT PERMUTATIONS
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

for q in range(N_QUBITS):

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
# Z EXPECTATION VALUES
# ============================================================

basis_indices = torch.arange(
    2 ** N_QUBITS,
    dtype=torch.long,
    device=device,
)

z_signs = []

for q in range(N_QUBITS):

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
        2.0 * bit.float()
    )


Z_SIGN_MATRIX = torch.stack(
    z_signs,
    dim=1,
)


# ============================================================
# SAMPLE LOGICAL PAULI CODES
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
# CHANNEL FOR ONE LOGICAL FAULT LOCATION
# ============================================================

def error_distribution_for_mode(
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

        return LOGICAL_CHANNELS[
            p
        ]["ideal_code"]


    if mode == "qec1":

        return LOGICAL_CHANNELS[
            p
        ]["qec1"]


    if mode == "qec3":

        return LOGICAL_CHANNELS[
            p
        ]["qec3"]


    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# LOGICAL VQC
#
# Same 8-logical-qubit classifier as before.
#
# At every modeled logical fault location, the physical
# [[5,1,3]] block + EC gadget is replaced by its empirically
# estimated logical Pauli channel.
# ============================================================

def run_logical_quantum_circuit(
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


    error_distribution = (
        error_distribution_for_mode(
            p,
            mode,
        )
    )


    encoding_errors = sample_pauli_codes(
        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
        ),
        error_distribution,
    )


    ry_errors = sample_pauli_codes(
        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
        ),
        error_distribution,
    )


    rz_errors = sample_pauli_codes(
        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
        ),
        error_distribution,
    )


    cnot_errors = sample_pauli_codes(
        (
            batch_size,
            N_UPLOADS,
            N_QUBITS,
            2,
        ),
        error_distribution,
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
        # Data upload
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
        # Trainable single-qubit gates
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
        # Logical CNOT ring
        # ----------------------------------------------------

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
# CIFAR-10 EVALUATION
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


            batch_size = (
                x.shape[0]
            )


            # Unprotected/ideal-code p=0 are exactly noiseless.
            #
            # QEC1/QEC3 at p=0 are NOT noiseless because their
            # stabilizer-extraction circuitry itself is noisy.

            if (
                p == 0
                and
                mode in [
                    "unprotected",
                    "ideal_code",
                ]
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
                    .mean(dim=1)
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
# MAIN EXPERIMENT
# ============================================================

print("\n======================================")
print("NOISY [[5,1,3]] STABILIZER EXPERIMENT")
print("======================================")

print("Trajectories:", N_TRAJECTORIES)
print("Seeds:", N_SEEDS)


MODES = [
    "unprotected",
    "ideal_code",
    "qec1",
    "qec3",
]


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


        seed_results = {}


        for mode_index, mode in enumerate(
            MODES
        ):

            # Reuse exact ideal result only for modes that
            # truly have no noise at p=0.

            if (
                p == 0
                and
                mode in [
                    "unprotected",
                    "ideal_code",
                ]
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
                        * mode_index
                    ),
                )


            seed_results[mode] = (
                accuracy,
                macro_f1,
                elapsed,
            )


        qec1_dist = (
            LOGICAL_CHANNELS[
                p
            ]["qec1"]
        )

        qec3_dist = (
            LOGICAL_CHANNELS[
                p
            ]["qec3"]
        )

        ideal_dist = (
            LOGICAL_CHANNELS[
                p
            ]["ideal_code"]
        )


        detailed_results.append({
            "p": p,
            "seed": base_seed,

            "ideal_code_logical_error":
                1.0 - ideal_dist[0],

            "qec1_logical_error":
                1.0 - qec1_dist[0],

            "qec3_logical_error":
                1.0 - qec3_dist[0],

            "qec1_logical_X":
                qec1_dist[1],

            "qec1_logical_Y":
                qec1_dist[2],

            "qec1_logical_Z":
                qec1_dist[3],

            "qec3_logical_X":
                qec3_dist[1],

            "qec3_logical_Y":
                qec3_dist[2],

            "qec3_logical_Z":
                qec3_dist[3],

            "unprotected_accuracy":
                seed_results[
                    "unprotected"
                ][0],

            "ideal_code_accuracy":
                seed_results[
                    "ideal_code"
                ][0],

            "qec1_accuracy":
                seed_results[
                    "qec1"
                ][0],

            "qec3_accuracy":
                seed_results[
                    "qec3"
                ][0],

            "unprotected_f1":
                seed_results[
                    "unprotected"
                ][1],

            "ideal_code_f1":
                seed_results[
                    "ideal_code"
                ][1],

            "qec1_f1":
                seed_results[
                    "qec1"
                ][1],

            "qec3_f1":
                seed_results[
                    "qec3"
                ][1],
        })


        print(
            f"Seed "
            f"{seed_index + 1}/{N_SEEDS} | "
            f"None="
            f"{seed_results['unprotected'][0]:.4f} | "
            f"Ideal5="
            f"{seed_results['ideal_code'][0]:.4f} | "
            f"QEC-1="
            f"{seed_results['qec1'][0]:.4f} | "
            f"QEC-3="
            f"{seed_results['qec3'][0]:.4f}"
        )


# ============================================================
# DETAILED CSV
# ============================================================

DETAILED_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_noisy_5qubit_stabilizer_detailed.csv",
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


    row_summary = {
        "p": p,

        "ideal_code_logical_error":
            subset[0][
                "ideal_code_logical_error"
            ],

        "qec1_logical_error":
            subset[0][
                "qec1_logical_error"
            ],

        "qec3_logical_error":
            subset[0][
                "qec3_logical_error"
            ],
    }


    for key in [
        "unprotected_accuracy",
        "ideal_code_accuracy",
        "qec1_accuracy",
        "qec3_accuracy",
    ]:

        values = np.array([
            row[key]
            for row in subset
        ])

        row_summary[
            key + "_mean"
        ] = values.mean()

        row_summary[
            key + "_std"
        ] = (
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )


    summary.append(
        row_summary
    )


SUMMARY_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_noisy_5qubit_stabilizer_summary.csv",
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
# PRINT SUMMARY
# ============================================================

print("\n======================================")
print("FINAL SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Unprotected\t"
    "Ideal [[5,1,3]]\t"
    "Noisy QEC-1\t"
    "Noisy QEC-3"
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

        f"{row['qec1_accuracy_mean']:.4f}"
        f" ± "
        f"{row['qec1_accuracy_std']:.4f}\t"

        f"{row['qec3_accuracy_mean']:.4f}"
        f" ± "
        f"{row['qec3_accuracy_std']:.4f}"
    )


print("\n======================================")
print("LOGICAL ERROR SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Ideal code pL\t"
    "Noisy QEC-1 pL\t"
    "Noisy QEC-3 pL"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['ideal_code_logical_error']:.6f}\t\t"

        f"{row['qec1_logical_error']:.6f}\t\t"

        f"{row['qec3_logical_error']:.6f}"
    )


# ============================================================
# PLOT ARRAYS
# ============================================================

p_values = np.array([
    row["p"]
    for row in summary
])


unprotected_mean = np.array([
    row[
        "unprotected_accuracy_mean"
    ]
    for row in summary
])

unprotected_std = np.array([
    row[
        "unprotected_accuracy_std"
    ]
    for row in summary
])


ideal_code_mean = np.array([
    row[
        "ideal_code_accuracy_mean"
    ]
    for row in summary
])

ideal_code_std = np.array([
    row[
        "ideal_code_accuracy_std"
    ]
    for row in summary
])


qec1_mean = np.array([
    row[
        "qec1_accuracy_mean"
    ]
    for row in summary
])

qec1_std = np.array([
    row[
        "qec1_accuracy_std"
    ]
    for row in summary
])


qec3_mean = np.array([
    row[
        "qec3_accuracy_mean"
    ]
    for row in summary
])

qec3_std = np.array([
    row[
        "qec3_accuracy_std"
    ]
    for row in summary
])


ideal_logical = np.array([
    row[
        "ideal_code_logical_error"
    ]
    for row in summary
])

qec1_logical = np.array([
    row[
        "qec1_logical_error"
    ]
    for row in summary
])

qec3_logical = np.array([
    row[
        "qec3_logical_error"
    ]
    for row in summary
])


# ============================================================
# FIGURE 1:
# CIFAR-10 ACCURACY
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.axhline(
    ideal_accuracy * 100,
    linestyle="--",
    label=(
        f"Ideal classifier "
        f"({ideal_accuracy * 100:.2f}%)"
    ),
)


plt.plot(
    p_values * 100,
    unprotected_mean * 100,
    marker="o",
    linewidth=2,
    label="Unprotected",
)


plt.plot(
    p_values * 100,
    ideal_code_mean * 100,
    marker="s",
    linewidth=2,
    label="Ideal [[5,1,3]] QEC",
)


plt.plot(
    p_values * 100,
    qec1_mean * 100,
    marker="^",
    linewidth=2,
    label="1-round noisy stabilizer EC",
)


plt.plot(
    p_values * 100,
    qec3_mean * 100,
    marker="D",
    linewidth=2,
    label="3-round noisy stabilizer EC",
)


plt.fill_between(
    p_values * 100,
    unprotected_mean * 100
    -
    unprotected_std * 100,
    unprotected_mean * 100
    +
    unprotected_std * 100,
    alpha=0.12,
)


plt.fill_between(
    p_values * 100,
    ideal_code_mean * 100
    -
    ideal_code_std * 100,
    ideal_code_mean * 100
    +
    ideal_code_std * 100,
    alpha=0.12,
)


plt.fill_between(
    p_values * 100,
    qec1_mean * 100
    -
    qec1_std * 100,
    qec1_mean * 100
    +
    qec1_std * 100,
    alpha=0.12,
)


plt.fill_between(
    p_values * 100,
    qec3_mean * 100
    -
    qec3_std * 100,
    qec3_mean * 100
    +
    qec3_std * 100,
    alpha=0.12,
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Noisy [[5,1,3]] Stabilizer QEC for CIFAR-10"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


ACCURACY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_noisy_5qubit_stabilizer_accuracy.png",
)


plt.savefig(
    ACCURACY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2:
# LOGICAL ERROR RATE
# ============================================================

plt.figure(
    figsize=(8, 5.8)
)


plt.plot(
    p_values * 100,
    p_values * 100,
    marker="o",
    linewidth=2,
    label="Unprotected physical p",
)


plt.plot(
    p_values * 100,
    ideal_logical * 100,
    marker="s",
    linewidth=2,
    label="Ideal [[5,1,3]]",
)


plt.plot(
    p_values * 100,
    qec1_logical * 100,
    marker="^",
    linewidth=2,
    label="Noisy stabilizer QEC-1",
)


plt.plot(
    p_values * 100,
    qec3_logical * 100,
    marker="D",
    linewidth=2,
    label="Noisy stabilizer QEC-3",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Residual Logical Pauli Error Probability (%)"
)

plt.title(
    "Logical Error with Noisy [[5,1,3]] Syndrome Extraction"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


LOGICAL_ERROR_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_noisy_stabilizer_logical_error.png",
)


plt.savefig(
    LOGICAL_ERROR_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3:
# QEC-3 MINUS QEC-1 CLASSIFICATION GAIN
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
    "QEC-3 − QEC-1 Accuracy (percentage points)"
)

plt.title(
    "Benefit or Cost of Repeating [[5,1,3]] Syndrome Extraction"
)

plt.grid(alpha=0.3)
plt.tight_layout()


GAIN_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_5qubit_qec3_vs_qec1_gain.png",
)


plt.savefig(
    GAIN_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 4:
# CIRCUIT-LEVEL LOGICAL PAULI COMPOSITION
# ============================================================

qec1_x = np.array([
    LOGICAL_CHANNELS[p][
        "qec1"
    ][1]
    for p in P_VALUES
])

qec1_y = np.array([
    LOGICAL_CHANNELS[p][
        "qec1"
    ][2]
    for p in P_VALUES
])

qec1_z = np.array([
    LOGICAL_CHANNELS[p][
        "qec1"
    ][3]
    for p in P_VALUES
])


qec3_x = np.array([
    LOGICAL_CHANNELS[p][
        "qec3"
    ][1]
    for p in P_VALUES
])

qec3_y = np.array([
    LOGICAL_CHANNELS[p][
        "qec3"
    ][2]
    for p in P_VALUES
])

qec3_z = np.array([
    LOGICAL_CHANNELS[p][
        "qec3"
    ][3]
    for p in P_VALUES
])


plt.figure(
    figsize=(8.5, 6)
)


plt.plot(
    p_values * 100,
    qec1_x * 100,
    marker="o",
    label="QEC-1 logical X",
)

plt.plot(
    p_values * 100,
    qec1_y * 100,
    marker="s",
    label="QEC-1 logical Y",
)

plt.plot(
    p_values * 100,
    qec1_z * 100,
    marker="^",
    label="QEC-1 logical Z",
)

plt.plot(
    p_values * 100,
    qec3_x * 100,
    marker="o",
    linestyle="--",
    label="QEC-3 logical X",
)

plt.plot(
    p_values * 100,
    qec3_y * 100,
    marker="s",
    linestyle="--",
    label="QEC-3 logical Y",
)

plt.plot(
    p_values * 100,
    qec3_z * 100,
    marker="^",
    linestyle="--",
    label="QEC-3 logical Z",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Residual Logical Pauli Probability (%)"
)

plt.title(
    "Residual Logical Pauli Channel from Noisy Stabilizer EC"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


PAULI_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_noisy_stabilizer_pauli_distribution.png",
)


plt.savefig(
    PAULI_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL FILE SUMMARY
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
    LOGICAL_ERROR_FIGURE
)

print(
    "QEC-3/QEC-1 figure:",
    GAIN_FIGURE
)

print(
    "Logical-Pauli figure:",
    PAULI_FIGURE
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "The five physical data qubits experience "
    "general depolarizing X/Y/Z errors."
)

print(
    "All four [[5,1,3]] stabilizers are explicitly "
    "measured with noisy ancilla-mediated circuits."
)

print(
    "Syndrome CNOTs, basis-changing H gates, "
    "ancilla preparation, ancilla measurement, "
    "and physical recovery are noisy."
)

print(
    "QEC-3 repeats all four stabilizer measurements "
    "three times and majority-votes each syndrome bit."
)

print(
    "A final ideal decoder is used only to characterize "
    "the residual logical I/X/Y/Z channel of the noisy "
    "error-correction gadget."
)

print(
    "The classifier is still simulated with eight "
    "logical qubits rather than a 40+ qubit statevector."
)

print(
    "The logical RY/RZ/CNOT implementation itself is "
    "still abstracted; this is a noisy stabilizer-EC "
    "model, not yet a complete fault-tolerant encoded "
    "implementation of the full VQC."
)

print(
    "The single-ancilla weight-4 stabilizer measurement "
    "used here is not itself a fully fault-tolerant "
    "flagged/verified extraction circuit; ancilla faults "
    "can propagate to multiple data qubits."
)
