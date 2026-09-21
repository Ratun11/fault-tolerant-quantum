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
# [[5,1,3]] FLAGGED SYNDROME-HISTORY DECODER FOR CIFAR-10
#
# NEXT STAGE AFTER:
#
#   cifar10_flagged_5qubit_ftec.py
#
# Goal:
# -----
# The previous flagged FTEC experiment showed that flags
# strongly reduce hook-error damage, but the logical-error
# floor is still significant because the QEC circuitry itself
# is noisy.
#
# In this script we study the next FTQC ingredient:
#
#                  SYNDROME HISTORY
#
# Instead of deciding from one syndrome or majority-voting
# three raw syndromes, we collect three full flagged rounds.
#
# Measured syndromes:
#
#   s^(1), s^(2), s^(3)
#
# Detection events:
#
#   d^(1) = s^(1)
#   d^(2) = s^(2) XOR s^(1)
#   d^(3) = s^(3) XOR s^(2)
#
# These changes contain temporal information that helps
# distinguish data faults from measurement faults.
#
# Because [[5,1,3]] is very small, we use a circuit-noise-
# calibrated maximum-likelihood lookup decoder rather than
# a large-code graph decoder.
#
# Decoder input:
#
#   12 detection-event bits
#       +
#   4 flag-summary bits
#
# = 16-bit history key.
#
# The flag-summary bit for stabilizer j is 1 if that
# stabilizer flagged in ANY of the three rounds.
#
# Decoder target:
# ----------------
# The final physical Pauli error is classified into one of
# 64 Pauli cosets:
#
#   16 final stabilizer syndromes
#       x
#    4 logical classes {I_L, X_L, Y_L, Z_L}
#
# The decoder learns the most likely final coset for every
# observed history key using synthetic calibration samples
# from the SAME circuit-level noise model.
#
# IMPORTANT:
# ----------
# This is a noise-model-specific ML lookup decoder.
# It is NOT trained on CIFAR labels and therefore does not
# use classification information.
#
# We compare:
#
#   1. Unprotected depolarizing channel
#   2. Ideal [[5,1,3]]
#   3. Previous adaptive flagged FTEC
#   4. Three-round raw-syndrome majority decoder
#   5. Three-round syndrome-history ML decoder
#
# The previous flagged channel is loaded from:
#
#   results/cifar10_flagged_5qubit_ftec_detailed.csv
#
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
# THREE-ROUND DECODER SETTINGS
# ============================================================

N_SYNDROME_ROUNDS = 3

# Training samples PER physical data-noise value.
#
# Seven p values -> 700,000 total calibration histories.
DECODER_TRAIN_SAMPLES_PER_P = 100_000

# Independent block samples used to evaluate each p after the
# decoder has been frozen.
BLOCK_EVAL_SAMPLES = 300_000


# ============================================================
# PHYSICAL DATA NOISE SWEEP
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
# CIRCUIT-LEVEL NOISE
#
# Keep identical to the preceding flagged-FTEC experiment.
# ============================================================

P_2Q = 0.005
P_1Q = 0.001
P_PREP = 0.001
P_MEAS = 0.010
P_RECOVERY = 0.005

# Keep False for direct continuity with the previous run.
#
# Set True in a later ablation if you want correction to be
# tracked entirely in software rather than physically applied.
USE_PAULI_FRAME_RECOVERY = False


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

PREVIOUS_FLAGGED_CSV = (
    "results/"
    "cifar10_flagged_5qubit_ftec_detailed.csv"
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
# LOAD CIFAR-10 FEATURES
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
# TRAINED QUANTUM CLASSIFIER
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

print(
    "Quantum weights:",
    model.quantum_weights.device,
)

print(
    "Classifier:",
    model.classifier[
        0
    ].weight.device,
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

LOGICAL_X_STRING = (
    "XXXXX"
)

LOGICAL_Z_STRING = (
    "ZZZZZ"
)

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
        len(
            pauli_string
        ),
        dtype=np.uint8,
    )

    z = np.zeros(
        len(
            pauli_string
        ),
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


    return (
        x,
        z,
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
        8
        *
        int(
            s[0]
        )
        +
        4
        *
        int(
            s[1]
        )
        +
        2
        *
        int(
            s[2]
        )
        +
        int(
            s[3]
        )
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


# ============================================================
# MINIMUM-WEIGHT DECODER
# ============================================================

DECODER = {}


for qubit in range(
    5
):

    for pauli_char in [
        "X",
        "Y",
        "Z",
    ]:

        error = [
            "I"
        ] * 5

        error[
            qubit
        ] = pauli_char


        error = (
            pauli_string_to_symplectic(
                "".join(
                    error
                )
            )
        )


        DECODER[
            syndrome_numpy(
                error
            )
        ] = error


assert len(
    DECODER
) == 15


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


for syn, correction in (
    DECODER.items()
):

    idx = (
        syndrome_tuple_to_index(
            syn
        )
    )


    DECODER_X_NP[
        idx
    ] = correction[
        0
    ]


    DECODER_Z_NP[
        idx
    ] = correction[
        1
    ]


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
    LOGICAL_X[
        0
    ],
    dtype=torch.bool,
    device=device,
)

LOGICAL_X_Z = torch.tensor(
    LOGICAL_X[
        1
    ],
    dtype=torch.bool,
    device=device,
)

LOGICAL_Z_X = torch.tensor(
    LOGICAL_Z[
        0
    ],
    dtype=torch.bool,
    device=device,
)

LOGICAL_Z_Z = torch.tensor(
    LOGICAL_Z[
        1
    ],
    dtype=torch.bool,
    device=device,
)


# ============================================================
# EXACT IDEAL [[5,1,3]] CHANNEL
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

        correction = (
            DECODER[
                syn
            ]
        )


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
    range(
        4
    ),
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


PATTERN_WEIGHTS = np.asarray(
    PATTERN_WEIGHTS,
)

PATTERN_LOGICAL_CLASSES = np.asarray(
    PATTERN_LOGICAL_CLASSES,
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
            (
                1.0
                -
                p
            )
            **
            (
                5
                -
                weight
            )
            *
            (
                p
                /
                3.0
            )
            **
            weight
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
# LOAD PREVIOUS FLAGGED-FTEC LOGICAL CHANNEL
#
# The previous script saved the same logical X/Y/Z channel in
# each of its five classifier-seed rows. We use the first row
# for each p.
# ============================================================

if not os.path.exists(
    PREVIOUS_FLAGGED_CSV
):

    raise FileNotFoundError(
        "\nRequired previous result not found:\n"
        f"{PREVIOUS_FLAGGED_CSV}\n\n"
        "Run cifar10_flagged_5qubit_ftec.py first."
    )


PREVIOUS_FLAGGED_CHANNELS = {}


with open(
    PREVIOUS_FLAGGED_CSV,
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


        if p in (
            PREVIOUS_FLAGGED_CHANNELS
        ):

            continue


        px = float(
            row[
                "flagged_logical_X"
            ]
        )

        py = float(
            row[
                "flagged_logical_Y"
            ]
        )

        pz = float(
            row[
                "flagged_logical_Z"
            ]
        )


        pi = (
            1.0
            -
            px
            -
            py
            -
            pz
        )


        PREVIOUS_FLAGGED_CHANNELS[
            p
        ] = np.array(
            [
                pi,
                px,
                py,
                pz,
            ],
            dtype=np.float64,
        )


print("\n======================================")
print("PREVIOUS FLAGGED-FTEC BASELINE")
print("======================================")


for p in P_VALUES:

    if p not in (
        PREVIOUS_FLAGGED_CHANNELS
    ):

        raise RuntimeError(
            f"Missing p={p} in "
            f"{PREVIOUS_FLAGGED_CSV}"
        )


    distribution = (
        PREVIOUS_FLAGGED_CHANNELS[
            p
        ]
    )


    print(
        f"p={p:.4f} | "
        f"previous flagged pL="
        f"{1.0 - distribution[0]:.6f}"
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


    return (
        x,
        z,
    )


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


    # Uniform over the 15 non-II Pauli products.
    pair_code = torch.randint(
        low=1,
        high=16,
        size=shape,
        device=device,
    )


    first = (
        pair_code
        //
        4
    )

    second = (
        pair_code
        %
        4
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
# PREPARED ANCILLA NOISE
# ============================================================

def prepare_zero_ancilla(
    samples,
):

    # X/Y act as the harmful bit flip on |0>.
    sx = bernoulli(
        (
            samples,
        ),
        2.0
        *
        P_PREP
        /
        3.0,
    )


    sz = torch.zeros(
        samples,
        dtype=torch.bool,
        device=device,
    )


    return (
        sx,
        sz,
    )


def prepare_plus_ancilla(
    samples,
):

    fx = torch.zeros(
        samples,
        dtype=torch.bool,
        device=device,
    )


    # Z/Y act as the harmful phase flip on |+>.
    fz = bernoulli(
        (
            samples,
        ),
        2.0
        *
        P_PREP
        /
        3.0,
    )


    return (
        fx,
        fz,
    )


# ============================================================
# NOISY PHYSICAL OPERATIONS
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


    fault_x, fault_z = (
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
        fault_x,
    )


    dz[
        :,
        q,
    ] = torch.logical_xor(
        dz[
            :,
            q,
        ],
        fault_z,
    )


def noisy_data_to_syndrome_cnot(
    dx,
    dz,
    q,
    sx,
    sz,
):

    data_x = dx[
        :,
        q,
    ].clone()

    data_z = dz[
        :,
        q,
    ].clone()

    syndrome_x = sx.clone()
    syndrome_z = sz.clone()


    # Ideal CNOT:
    # data = control
    # syndrome = target

    dx[
        :,
        q,
    ] = data_x


    dz[
        :,
        q,
    ] = torch.logical_xor(
        data_z,
        syndrome_z,
    )


    sx = torch.logical_xor(
        syndrome_x,
        data_x,
    )


    sz = syndrome_z


    (
        fault_x_data,
        fault_z_data,
        fault_x_syndrome,
        fault_z_syndrome,
    ) = (
        sample_two_qubit_depolarizing(
            sx.shape,
            P_2Q,
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
        fault_x_data,
    )


    dz[
        :,
        q,
    ] = torch.logical_xor(
        dz[
            :,
            q,
        ],
        fault_z_data,
    )


    sx = torch.logical_xor(
        sx,
        fault_x_syndrome,
    )


    sz = torch.logical_xor(
        sz,
        fault_z_syndrome,
    )


    return (
        sx,
        sz,
    )


def noisy_flag_to_syndrome_cnot(
    fx,
    fz,
    sx,
    sz,
):

    flag_x = fx.clone()
    flag_z = fz.clone()

    syndrome_x = sx.clone()
    syndrome_z = sz.clone()


    # flag = control
    # syndrome = target

    fx = flag_x


    fz = torch.logical_xor(
        flag_z,
        syndrome_z,
    )


    sx = torch.logical_xor(
        syndrome_x,
        flag_x,
    )


    sz = syndrome_z


    (
        fault_x_flag,
        fault_z_flag,
        fault_x_syndrome,
        fault_z_syndrome,
    ) = (
        sample_two_qubit_depolarizing(
            sx.shape,
            P_2Q,
        )
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
        fault_x_syndrome,
    )


    sz = torch.logical_xor(
        sz,
        fault_z_syndrome,
    )


    return (
        fx,
        fz,
        sx,
        sz,
    )


# ============================================================
# ONE FLAGGED STABILIZER MEASUREMENT
# ============================================================

def measure_stabilizer_flagged(
    dx,
    dz,
    stabilizer_string,
):

    samples = dx.shape[
        0
    ]


    sx, sz = (
        prepare_zero_ancilla(
            samples
        )
    )


    fx, fz = (
        prepare_plus_ancilla(
            samples
        )
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
            stabilizer_string[
                q
            ]
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


        # Flag coupling after data interaction 1.
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


        # Flag coupling after data interaction 3.
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


    # Z-basis syndrome measurement.
    measured_syndrome = sx.clone()


    measured_syndrome = (
        torch.logical_xor(
            measured_syndrome,
            bernoulli(
                measured_syndrome.shape,
                P_MEAS,
            ),
        )
    )


    # X-basis flag measurement.
    measured_flag = fz.clone()


    measured_flag = (
        torch.logical_xor(
            measured_flag,
            bernoulli(
                measured_flag.shape,
                P_MEAS,
            ),
        )
    )


    return (
        dx,
        dz,
        measured_syndrome,
        measured_flag,
    )


# ============================================================
# ONE COMPLETE FLAGGED SYNDROME ROUND
#
# Measures all 4 stabilizers.
# ============================================================

def extract_flagged_round(
    dx,
    dz,
):

    syndrome_bits = []
    flag_bits = []


    for stabilizer_string in (
        STABILIZER_STRINGS
    ):

        (
            dx,
            dz,
            measured_syndrome,
            measured_flag,
        ) = (
            measure_stabilizer_flagged(
                dx,
                dz,
                stabilizer_string,
            )
        )


        syndrome_bits.append(
            measured_syndrome
        )


        flag_bits.append(
            measured_flag
        )


    syndrome_bits = torch.stack(
        syndrome_bits,
        dim=1,
    )


    flag_bits = torch.stack(
        flag_bits,
        dim=1,
    )


    return (
        dx,
        dz,
        syndrome_bits,
        flag_bits,
    )


# ============================================================
# THREE-ROUND HISTORY SIMULATION
# ============================================================

def simulate_three_round_history(
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
        sample_single_qubit_depolarizing(
            (
                samples,
                5,
            ),
            p_data,
        )
    )


    syndrome_history = []
    flag_history = []


    for _ in range(
        N_SYNDROME_ROUNDS
    ):

        (
            dx,
            dz,
            syndrome_bits,
            flag_bits,
        ) = (
            extract_flagged_round(
                dx,
                dz,
            )
        )


        syndrome_history.append(
            syndrome_bits
        )


        flag_history.append(
            flag_bits
        )


    syndrome_history = torch.stack(
        syndrome_history,
        dim=1,
    )


    flag_history = torch.stack(
        flag_history,
        dim=1,
    )


    return (
        dx,
        dz,
        syndrome_history,
        flag_history,
    )


# ============================================================
# TRUE SYNDROME OF A PHYSICAL PAULI FRAME
# ============================================================

def true_syndrome_torch(
    dx,
    dz,
):

    batch_size = dx.shape[
        0
    ]


    syndrome_bits = torch.zeros(
        batch_size,
        4,
        dtype=torch.bool,
        device=device,
    )


    for q in range(
        5
    ):

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
        8
        *
        s[
            :,
            0,
        ]
        +
        4
        *
        s[
            :,
            1,
        ]
        +
        2
        *
        s[
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
# LOGICAL CLASS AFTER CANONICAL IDEAL DECODING
# ============================================================

def residual_logical_class(
    residual_x,
    residual_z,
):

    logical_x_bit = torch.zeros(
        residual_x.shape[
            0
        ],
        dtype=torch.bool,
        device=device,
    )


    logical_z_bit = torch.zeros(
        residual_x.shape[
            0
        ],
        dtype=torch.bool,
        device=device,
    )


    for q in range(
        5
    ):

        logical_x_bit = torch.logical_xor(
            logical_x_bit,
            torch.logical_and(
                residual_x[
                    :,
                    q,
                ],
                LOGICAL_Z_Z[
                    q
                ],
            ),
        )


        logical_x_bit = torch.logical_xor(
            logical_x_bit,
            torch.logical_and(
                residual_z[
                    :,
                    q,
                ],
                LOGICAL_Z_X[
                    q
                ],
            ),
        )


        logical_z_bit = torch.logical_xor(
            logical_z_bit,
            torch.logical_and(
                residual_x[
                    :,
                    q,
                ],
                LOGICAL_X_Z[
                    q
                ],
            ),
        )


        logical_z_bit = torch.logical_xor(
            logical_z_bit,
            torch.logical_and(
                residual_z[
                    :,
                    q,
                ],
                LOGICAL_X_X[
                    q
                ],
            ),
        )


    logical_class = torch.zeros(
        residual_x.shape[
            0
        ],
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
# PHYSICAL FRAME -> 64-CLASS COSET LABEL
#
# label = 4 * syndrome_index + logical_class
#
# syndrome_index: 0..15
# logical_class:  0..3
# ============================================================

def frame_to_coset_class(
    dx,
    dz,
):

    syndrome_bits = (
        true_syndrome_torch(
            dx,
            dz,
        )
    )


    syndrome_index = (
        syndrome_bits_to_index_torch(
            syndrome_bits
        )
    )


    canonical_x = (
        DECODER_X[
            syndrome_index
        ]
    )


    canonical_z = (
        DECODER_Z[
            syndrome_index
        ]
    )


    residual_x = (
        torch.logical_xor(
            dx,
            canonical_x,
        )
    )


    residual_z = (
        torch.logical_xor(
            dz,
            canonical_z,
        )
    )


    logical_class = (
        residual_logical_class(
            residual_x,
            residual_z,
        )
    )


    return (
        4
        *
        syndrome_index
        +
        logical_class
    )


# ============================================================
# HISTORY KEY
#
# Input:
# syndrome_history [N,3,4]
# flag_history     [N,3,4]
#
# Detection events:
#
# d0 = s0
# d1 = s1 XOR s0
# d2 = s2 XOR s1
#
# Flag summary:
#
# f_any[j] = OR_r flag[r,j]
#
# Total:
# 12 + 4 = 16 bits
#
# key in [0, 65535]
# ============================================================

def history_to_key(
    syndrome_history,
    flag_history,
):

    d0 = syndrome_history[
        :,
        0,
        :,
    ]


    d1 = torch.logical_xor(
        syndrome_history[
            :,
            1,
            :,
        ],
        syndrome_history[
            :,
            0,
            :,
        ],
    )


    d2 = torch.logical_xor(
        syndrome_history[
            :,
            2,
            :,
        ],
        syndrome_history[
            :,
            1,
            :,
        ],
    )


    flag_any = torch.any(
        flag_history,
        dim=1,
    )


    bits = torch.cat(
        [
            d0,
            d1,
            d2,
            flag_any,
        ],
        dim=1,
    ).to(
        torch.long
    )


    key = torch.zeros(
        bits.shape[
            0
        ],
        dtype=torch.long,
        device=device,
    )


    for bit_index in range(
        16
    ):

        key = (
            key
            |
            (
                bits[
                    :,
                    bit_index,
                ]
                <<
                bit_index
            )
        )


    return key


# ============================================================
# FALLBACK DECODER FOR UNSEEN HISTORY KEYS
#
# Reconstruct final measured syndrome:
#
# s3 = d0 XOR d1 XOR d2
#
# Then use standard minimum-weight correction and assume
# logical class I.
# ============================================================

def fallback_class_for_all_keys():

    keys = np.arange(
        65536,
        dtype=np.uint32,
    )


    final_bits = np.zeros(
        (
            65536,
            4,
        ),
        dtype=np.uint8,
    )


    for stabilizer_index in range(
        4
    ):

        d0 = (
            keys
            >>
            stabilizer_index
        ) & 1


        d1 = (
            keys
            >>
            (
                4
                +
                stabilizer_index
            )
        ) & 1


        d2 = (
            keys
            >>
            (
                8
                +
                stabilizer_index
            )
        ) & 1


        final_bits[
            :,
            stabilizer_index,
        ] = (
            d0
            ^
            d1
            ^
            d2
        )


    syndrome_index = (
        8
        *
        final_bits[
            :,
            0,
        ]
        +
        4
        *
        final_bits[
            :,
            1,
        ]
        +
        2
        *
        final_bits[
            :,
            2,
        ]
        +
        final_bits[
            :,
            3,
        ]
    )


    return (
        4
        *
        syndrome_index
    ).astype(
        np.uint8
    )


# ============================================================
# BUILD MAXIMUM-LIKELIHOOD HISTORY DECODER
#
# Counts:
#
# history_key x final_coset_class
#
# = 65536 x 64
#
# ~4.2M integer counters.
# ============================================================

print("\n======================================")
print("TRAINING SYNDROME-HISTORY DECODER")
print("======================================")

print(
    "Rounds:",
    N_SYNDROME_ROUNDS,
)

print(
    "Samples per p:",
    DECODER_TRAIN_SAMPLES_PER_P,
)

print(
    "Total calibration samples:",
    (
        DECODER_TRAIN_SAMPLES_PER_P
        *
        len(
            P_VALUES
        )
    ),
)


history_class_counts = np.zeros(
    (
        65536,
        64,
    ),
    dtype=np.uint32,
)


decoder_training_start = (
    time.time()
)


for p_index, p in enumerate(
    P_VALUES
):

    (
        train_dx,
        train_dz,
        train_syndrome_history,
        train_flag_history,
    ) = simulate_three_round_history(
        p_data=p,
        samples=(
            DECODER_TRAIN_SAMPLES_PER_P
        ),
        random_seed=(
            SEED
            +
            1_000_000
            +
            1000
            *
            p_index
        ),
    )


    train_keys = (
        history_to_key(
            train_syndrome_history,
            train_flag_history,
        )
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )


    train_classes = (
        frame_to_coset_class(
            train_dx,
            train_dz,
        )
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )


    np.add.at(
        history_class_counts,
        (
            train_keys,
            train_classes,
        ),
        1,
    )


    print(
        f"p={p:.4f} | "
        f"calibration samples="
        f"{DECODER_TRAIN_SAMPLES_PER_P}"
    )


history_totals = (
    history_class_counts.sum(
        axis=1
    )
)


history_decoder_table = (
    np.argmax(
        history_class_counts,
        axis=1,
    )
    .astype(
        np.uint8
    )
)


fallback_table = (
    fallback_class_for_all_keys()
)


unseen_mask = (
    history_totals
    ==
    0
)


history_decoder_table[
    unseen_mask
] = fallback_table[
    unseen_mask
]


seen_history_keys = int(
    np.count_nonzero(
        ~unseen_mask
    )
)


training_correct = (
    history_class_counts[
        np.arange(
            65536
        ),
        history_decoder_table.astype(
            np.int64
        ),
    ].sum()
)


training_total = (
    history_class_counts.sum()
)


training_ml_accuracy = (
    float(
        training_correct
    )
    /
    float(
        training_total
    )
)


print(
    "\nObserved history keys:",
    seen_history_keys,
    "/ 65536",
)

print(
    "Calibration ML coset accuracy:",
    f"{training_ml_accuracy:.4f}",
)

print(
    "Decoder build time:",
    f"{time.time() - decoder_training_start:.1f}",
    "sec",
)


HISTORY_DECODER_TABLE = torch.tensor(
    history_decoder_table,
    dtype=torch.long,
    device=device,
)


# ============================================================
# CLASS -> PHYSICAL CORRECTION
#
# coset class:
#
#   syndrome canonical representative
#       *
#   logical representative
#
# Pauli phases are ignored.
# ============================================================

def coset_class_to_correction(
    coset_class,
):

    syndrome_index = (
        coset_class
        //
        4
    )


    logical_class = (
        coset_class
        %
        4
    )


    correction_x = (
        DECODER_X[
            syndrome_index
        ]
        .clone()
    )


    correction_z = (
        DECODER_Z[
            syndrome_index
        ]
        .clone()
    )


    logical_x_mask = torch.logical_or(
        logical_class == 1,
        logical_class == 2,
    )


    logical_z_mask = torch.logical_or(
        logical_class == 2,
        logical_class == 3,
    )


    correction_x = torch.logical_xor(
        correction_x,
        torch.logical_and(
            logical_x_mask.unsqueeze(
                1
            ),
            LOGICAL_X_X.unsqueeze(
                0
            ),
        ),
    )


    correction_z = torch.logical_xor(
        correction_z,
        torch.logical_and(
            logical_x_mask.unsqueeze(
                1
            ),
            LOGICAL_X_Z.unsqueeze(
                0
            ),
        ),
    )


    correction_x = torch.logical_xor(
        correction_x,
        torch.logical_and(
            logical_z_mask.unsqueeze(
                1
            ),
            LOGICAL_Z_X.unsqueeze(
                0
            ),
        ),
    )


    correction_z = torch.logical_xor(
        correction_z,
        torch.logical_and(
            logical_z_mask.unsqueeze(
                1
            ),
            LOGICAL_Z_Z.unsqueeze(
                0
            ),
        ),
    )


    return (
        correction_x,
        correction_z,
    )


# ============================================================
# APPLY RECOVERY
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

        return (
            dx,
            dz,
        )


    recovery_mask = torch.logical_or(
        correction_x,
        correction_z,
    )


    fault_x, fault_z = (
        sample_single_qubit_depolarizing(
            dx.shape,
            P_RECOVERY,
        )
    )


    fault_x = torch.logical_and(
        fault_x,
        recovery_mask,
    )


    fault_z = torch.logical_and(
        fault_z,
        recovery_mask,
    )


    dx = torch.logical_xor(
        dx,
        fault_x,
    )


    dz = torch.logical_xor(
        dz,
        fault_z,
    )


    return (
        dx,
        dz,
    )


# ============================================================
# FINAL LOGICAL DISTRIBUTION
#
# An ideal final decoder is used ONLY for channel
# characterization, just as in the earlier experiments.
# ============================================================

def final_logical_distribution(
    dx,
    dz,
):

    true_syndrome = (
        true_syndrome_torch(
            dx,
            dz,
        )
    )


    syndrome_index = (
        syndrome_bits_to_index_torch(
            true_syndrome
        )
    )


    canonical_x = (
        DECODER_X[
            syndrome_index
        ]
    )


    canonical_z = (
        DECODER_Z[
            syndrome_index
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


    logical_class = (
        residual_logical_class(
            residual_x,
            residual_z,
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
# EVALUATE DECODERS ON INDEPENDENT HISTORY SAMPLES
#
# Returns:
#
#   majority raw-syndrome decoder
#   last-round raw-syndrome decoder
#   history ML decoder
#
# plus direct coset-prediction accuracy.
# ============================================================

def evaluate_history_decoders(
    p_data,
    samples,
    random_seed,
):

    (
        dx,
        dz,
        syndrome_history,
        flag_history,
    ) = simulate_three_round_history(
        p_data=p_data,
        samples=samples,
        random_seed=random_seed,
    )


    true_coset_class = (
        frame_to_coset_class(
            dx,
            dz,
        )
    )


    # ========================================================
    # 1) RAW-SYNDROME MAJORITY
    # ========================================================

    majority_syndrome = (
        syndrome_history
        .sum(
            dim=1
        )
        >= 2
    )


    majority_index = (
        syndrome_bits_to_index_torch(
            majority_syndrome
        )
    )


    majority_correction_x = (
        DECODER_X[
            majority_index
        ]
    )


    majority_correction_z = (
        DECODER_Z[
            majority_index
        ]
    )


    majority_dx = dx.clone()
    majority_dz = dz.clone()


    torch.manual_seed(
        random_seed
        +
        100_000
    )

    torch.cuda.manual_seed_all(
        random_seed
        +
        100_000
    )


    (
        majority_dx,
        majority_dz,
    ) = apply_recovery(
        majority_dx,
        majority_dz,
        majority_correction_x,
        majority_correction_z,
    )


    majority_distribution = (
        final_logical_distribution(
            majority_dx,
            majority_dz,
        )
    )


    # ========================================================
    # 2) LAST ROUND ONLY
    # ========================================================

    last_syndrome = (
        syndrome_history[
            :,
            -1,
            :,
        ]
    )


    last_index = (
        syndrome_bits_to_index_torch(
            last_syndrome
        )
    )


    last_correction_x = (
        DECODER_X[
            last_index
        ]
    )


    last_correction_z = (
        DECODER_Z[
            last_index
        ]
    )


    last_dx = dx.clone()
    last_dz = dz.clone()


    torch.manual_seed(
        random_seed
        +
        200_000
    )

    torch.cuda.manual_seed_all(
        random_seed
        +
        200_000
    )


    (
        last_dx,
        last_dz,
    ) = apply_recovery(
        last_dx,
        last_dz,
        last_correction_x,
        last_correction_z,
    )


    last_distribution = (
        final_logical_distribution(
            last_dx,
            last_dz,
        )
    )


    # ========================================================
    # 3) HISTORY ML DECODER
    # ========================================================

    history_keys = (
        history_to_key(
            syndrome_history,
            flag_history,
        )
    )


    predicted_coset_class = (
        HISTORY_DECODER_TABLE[
            history_keys
        ]
    )


    coset_prediction_accuracy = (
        (
            predicted_coset_class
            ==
            true_coset_class
        )
        .float()
        .mean()
        .item()
    )


    (
        history_correction_x,
        history_correction_z,
    ) = coset_class_to_correction(
        predicted_coset_class
    )


    history_dx = dx.clone()
    history_dz = dz.clone()


    torch.manual_seed(
        random_seed
        +
        300_000
    )

    torch.cuda.manual_seed_all(
        random_seed
        +
        300_000
    )


    (
        history_dx,
        history_dz,
    ) = apply_recovery(
        history_dx,
        history_dz,
        history_correction_x,
        history_correction_z,
    )


    history_distribution = (
        final_logical_distribution(
            history_dx,
            history_dz,
        )
    )


    # ========================================================
    # HISTORY STATISTICS
    # ========================================================

    any_flag = torch.any(
        flag_history,
        dim=(
            1,
            2,
        ),
    )


    flag_cycle_rate = (
        any_flag
        .float()
        .mean()
        .item()
    )


    syndrome_changes = torch.logical_xor(
        syndrome_history[
            :,
            1:,
            :,
        ],
        syndrome_history[
            :,
            :-1,
            :,
        ],
    )


    any_temporal_change = torch.any(
        syndrome_changes,
        dim=(
            1,
            2,
        ),
    )


    temporal_change_rate = (
        any_temporal_change
        .float()
        .mean()
        .item()
    )


    return {
        "majority":
            majority_distribution,

        "last":
            last_distribution,

        "history":
            history_distribution,

        "coset_prediction_accuracy":
            coset_prediction_accuracy,

        "flag_cycle_rate":
            flag_cycle_rate,

        "temporal_change_rate":
            temporal_change_rate,
    }


# ============================================================
# EVALUATE LOGICAL CHANNELS
# ============================================================

print("\n======================================")
print("INDEPENDENT DECODER EVALUATION")
print("======================================")

print(
    "Evaluation samples per p:",
    BLOCK_EVAL_SAMPLES,
)


LOGICAL_CHANNELS = {}


decoder_metrics = []


for p_index, p in enumerate(
    P_VALUES
):

    result = (
        evaluate_history_decoders(
            p_data=p,
            samples=(
                BLOCK_EVAL_SAMPLES
            ),
            random_seed=(
                SEED
                +
                5_000_000
                +
                1000
                *
                p_index
            ),
        )
    )


    ideal_distribution = (
        ideal_five_qubit_logical_distribution(
            p
        )
    )


    previous_flagged = (
        PREVIOUS_FLAGGED_CHANNELS[
            p
        ]
    )


    LOGICAL_CHANNELS[
        p
    ] = {
        "ideal_code":
            ideal_distribution,

        "previous_flagged":
            previous_flagged,

        "majority3":
            result[
                "majority"
            ],

        "last3":
            result[
                "last"
            ],

        "history3":
            result[
                "history"
            ],
    }


    decoder_metrics.append(
        {
            "p":
                p,

            "coset_prediction_accuracy":
                result[
                    "coset_prediction_accuracy"
                ],

            "flag_cycle_rate":
                result[
                    "flag_cycle_rate"
                ],

            "temporal_change_rate":
                result[
                    "temporal_change_rate"
                ],
        }
    )


    print(
        "\n--------------------------------------"
    )

    print(
        f"p={p:.4f}"
    )


    print(
        f"Previous flagged pL = "
        f"{1.0 - previous_flagged[0]:.6f}"
    )


    print(
        f"3-round majority pL = "
        f"{1.0 - result['majority'][0]:.6f}"
    )


    print(
        f"3-round last-only pL = "
        f"{1.0 - result['last'][0]:.6f}"
    )


    print(
        f"3-round history pL = "
        f"{1.0 - result['history'][0]:.6f}"
    )


    print(
        f"Coset prediction accuracy = "
        f"{result['coset_prediction_accuracy']:.4f}"
    )


    print(
        f"Any flag during 3 rounds = "
        f"{result['flag_cycle_rate']:.4f}"
    )


    print(
        f"Any temporal syndrome change = "
        f"{result['temporal_change_rate']:.4f}"
    )


# ============================================================
# SAVE DECODER CHANNEL SUMMARY
# ============================================================

DECODER_CHANNEL_CSV = os.path.join(
    RESULT_DIR,
    "five_qubit_syndrome_history_channels.csv",
)


with open(
    DECODER_CHANNEL_CSV,
    "w",
    newline="",
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow(
        [
            "p",
            "ideal_logical_error",
            "previous_flagged_logical_error",
            "majority3_logical_error",
            "last3_logical_error",
            "history3_logical_error",
            "history_coset_prediction_accuracy",
            "three_round_flag_cycle_rate",
            "temporal_syndrome_change_rate",
        ]
    )


    for metric in decoder_metrics:

        p = metric[
            "p"
        ]


        writer.writerow(
            [
                p,

                1.0
                -
                LOGICAL_CHANNELS[
                    p
                ][
                    "ideal_code"
                ][0],

                1.0
                -
                LOGICAL_CHANNELS[
                    p
                ][
                    "previous_flagged"
                ][0],

                1.0
                -
                LOGICAL_CHANNELS[
                    p
                ][
                    "majority3"
                ][0],

                1.0
                -
                LOGICAL_CHANNELS[
                    p
                ][
                    "last3"
                ][0],

                1.0
                -
                LOGICAL_CHANNELS[
                    p
                ][
                    "history3"
                ][0],

                metric[
                    "coset_prediction_accuracy"
                ],

                metric[
                    "flag_cycle_rate"
                ],

                metric[
                    "temporal_change_rate"
                ],
            ]
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
# SAMPLE LOGICAL PAULI CHANNEL
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
# CHANNEL BY MODE
# ============================================================

def channel_for_mode(
    p,
    mode,
):

    if mode == "unprotected":

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


    if mode == "ideal_code":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "ideal_code"
            ]
        )


    if mode == "previous_flagged":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "previous_flagged"
            ]
        )


    if mode == "majority3":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "majority3"
            ]
        )


    if mode == "history3":

        return (
            LOGICAL_CHANNELS[
                p
            ][
                "history3"
            ]
        )


    raise ValueError(
        f"Unknown mode: "
        f"{mode}"
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

    batch_size = (
        inputs.shape[
            0
        ]
    )


    state = initial_state(
        batch_size
    )


    error_distribution = (
        channel_for_mode(
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
            error_distribution,
        )
    )


    ry_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            error_distribution,
        )
    )


    rz_errors = (
        sample_pauli_codes(
            (
                batch_size,
                N_UPLOADS,
                N_QUBITS,
            ),
            error_distribution,
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
            error_distribution,
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
                    start
                    +
                    q,
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

    start_time = (
        time.time()
    )


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


            logits = (
                model.classifier(
                    quantum_features
                )
            )


            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
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
# IDEAL CLASSIFIER
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
    "Ideal accuracy =",
    f"{ideal_accuracy:.4f}",
)

print(
    "Ideal Macro-F1 =",
    f"{ideal_f1:.4f}",
)


# ============================================================
# CIFAR-10 EXPERIMENT
# ============================================================

MODES = [
    "unprotected",
    "ideal_code",
    "previous_flagged",
    "majority3",
    "history3",
]


detailed_results = []


print("\n======================================")
print("SYNDROME-HISTORY CIFAR-10 EXPERIMENT")
print("======================================")

print(
    "Trajectories:",
    N_TRAJECTORIES,
)

print(
    "Classifier seeds:",
    N_SEEDS,
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
            *
            p_index
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
                        100_000
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


        metric = (
            decoder_metrics[
                p_index
            ]
        )


        detailed_results.append(
            {
                "p":
                    p,

                "seed":
                    base_seed,

                "ideal_logical_error":
                    1.0
                    -
                    LOGICAL_CHANNELS[
                        p
                    ][
                        "ideal_code"
                    ][0],

                "previous_flagged_logical_error":
                    1.0
                    -
                    LOGICAL_CHANNELS[
                        p
                    ][
                        "previous_flagged"
                    ][0],

                "majority3_logical_error":
                    1.0
                    -
                    LOGICAL_CHANNELS[
                        p
                    ][
                        "majority3"
                    ][0],

                "history3_logical_error":
                    1.0
                    -
                    LOGICAL_CHANNELS[
                        p
                    ][
                        "history3"
                    ][0],

                "history_coset_prediction_accuracy":
                    metric[
                        "coset_prediction_accuracy"
                    ],

                "temporal_syndrome_change_rate":
                    metric[
                        "temporal_change_rate"
                    ],

                "three_round_flag_cycle_rate":
                    metric[
                        "flag_cycle_rate"
                    ],

                "unprotected_accuracy":
                    mode_results[
                        "unprotected"
                    ][0],

                "ideal_code_accuracy":
                    mode_results[
                        "ideal_code"
                    ][0],

                "previous_flagged_accuracy":
                    mode_results[
                        "previous_flagged"
                    ][0],

                "majority3_accuracy":
                    mode_results[
                        "majority3"
                    ][0],

                "history3_accuracy":
                    mode_results[
                        "history3"
                    ][0],

                "unprotected_f1":
                    mode_results[
                        "unprotected"
                    ][1],

                "ideal_code_f1":
                    mode_results[
                        "ideal_code"
                    ][1],

                "previous_flagged_f1":
                    mode_results[
                        "previous_flagged"
                    ][1],

                "majority3_f1":
                    mode_results[
                        "majority3"
                    ][1],

                "history3_f1":
                    mode_results[
                        "history3"
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

            f"FlagPrev="
            f"{mode_results['previous_flagged'][0]:.4f} | "

            f"Majority3="
            f"{mode_results['majority3'][0]:.4f} | "

            f"History3="
            f"{mode_results['history3'][0]:.4f}"
        )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

DETAILED_CSV = os.path.join(
    RESULT_DIR,
    "cifar10_syndrome_history_detailed.csv",
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
        if row[
            "p"
        ] == p
    ]


    summary_row = {
        "p":
            p,

        "ideal_logical_error":
            subset[
                0
            ][
                "ideal_logical_error"
            ],

        "previous_flagged_logical_error":
            subset[
                0
            ][
                "previous_flagged_logical_error"
            ],

        "majority3_logical_error":
            subset[
                0
            ][
                "majority3_logical_error"
            ],

        "history3_logical_error":
            subset[
                0
            ][
                "history3_logical_error"
            ],

        "history_coset_prediction_accuracy":
            subset[
                0
            ][
                "history_coset_prediction_accuracy"
            ],

        "temporal_syndrome_change_rate":
            subset[
                0
            ][
                "temporal_syndrome_change_rate"
            ],

        "three_round_flag_cycle_rate":
            subset[
                0
            ][
                "three_round_flag_cycle_rate"
            ],
    }


    for key in [
        "unprotected_accuracy",
        "ideal_code_accuracy",
        "previous_flagged_accuracy",
        "majority3_accuracy",
        "history3_accuracy",
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
    "cifar10_syndrome_history_summary.csv",
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
    "Ideal5\t\t"
    "FlagPrev\t"
    "Majority3\t"
    "History3"
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

        f"{row['previous_flagged_accuracy_mean']:.4f}"
        f" ± "
        f"{row['previous_flagged_accuracy_std']:.4f}\t"

        f"{row['majority3_accuracy_mean']:.4f}"
        f" ± "
        f"{row['majority3_accuracy_std']:.4f}\t"

        f"{row['history3_accuracy_mean']:.4f}"
        f" ± "
        f"{row['history3_accuracy_std']:.4f}"
    )


print("\n======================================")
print("LOGICAL ERROR SUMMARY")
print("======================================")

print(
    "p\t\t"
    "Ideal pL\t"
    "FlagPrev pL\t"
    "Majority3 pL\t"
    "History3 pL"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['ideal_logical_error']:.6f}\t"

        f"{row['previous_flagged_logical_error']:.6f}\t"

        f"{row['majority3_logical_error']:.6f}\t\t"

        f"{row['history3_logical_error']:.6f}"
    )


print("\n======================================")
print("HISTORY DECODER DIAGNOSTICS")
print("======================================")

print(
    "p\t\t"
    "Coset accuracy\t"
    "Temporal change\t"
    "Any flag"
)


for row in summary:

    print(
        f"{row['p']:.4f}\t\t"

        f"{row['history_coset_prediction_accuracy']:.4f}\t\t"

        f"{row['temporal_syndrome_change_rate']:.4f}\t\t"

        f"{row['three_round_flag_cycle_rate']:.4f}"
    )


# ============================================================
# PLOT ARRAYS
# ============================================================

p_values = np.array(
    [
        row[
            "p"
        ]
        for row in summary
    ]
)


def get_summary_array(
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


none_mean = get_summary_array(
    "unprotected_accuracy_mean"
)

none_std = get_summary_array(
    "unprotected_accuracy_std"
)


ideal_mean = get_summary_array(
    "ideal_code_accuracy_mean"
)

ideal_std = get_summary_array(
    "ideal_code_accuracy_std"
)


previous_flagged_mean = (
    get_summary_array(
        "previous_flagged_accuracy_mean"
    )
)

previous_flagged_std = (
    get_summary_array(
        "previous_flagged_accuracy_std"
    )
)


majority_mean = get_summary_array(
    "majority3_accuracy_mean"
)

majority_std = get_summary_array(
    "majority3_accuracy_std"
)


history_mean = get_summary_array(
    "history3_accuracy_mean"
)

history_std = get_summary_array(
    "history3_accuracy_std"
)


ideal_pl = get_summary_array(
    "ideal_logical_error"
)

previous_flagged_pl = (
    get_summary_array(
        "previous_flagged_logical_error"
    )
)

majority_pl = get_summary_array(
    "majority3_logical_error"
)

history_pl = get_summary_array(
    "history3_logical_error"
)


coset_accuracy = get_summary_array(
    "history_coset_prediction_accuracy"
)

temporal_change = get_summary_array(
    "temporal_syndrome_change_rate"
)

flag_cycle_rate = get_summary_array(
    "three_round_flag_cycle_rate"
)


# ============================================================
# FIGURE 1:
# CIFAR ACCURACY
# ============================================================

plt.figure(
    figsize=(
        9.5,
        6.2,
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


curves = [
    (
        none_mean,
        none_std,
        "o",
        "Unprotected",
    ),
    (
        ideal_mean,
        ideal_std,
        "s",
        "Ideal [[5,1,3]]",
    ),
    (
        previous_flagged_mean,
        previous_flagged_std,
        "^",
        "Previous flagged FTEC",
    ),
    (
        majority_mean,
        majority_std,
        "D",
        "3-round raw majority",
    ),
    (
        history_mean,
        history_std,
        "P",
        "3-round history decoder",
    ),
]


for mean, std, marker, label in curves:

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

        alpha=0.08,
    )


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "CIFAR-10 Test Accuracy (%)"
)

plt.title(
    "Syndrome-History Decoding for Flagged [[5,1,3]] QEC"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


ACCURACY_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_syndrome_history_accuracy.png",
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
        8.5,
        5.8,
    )
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
    previous_flagged_pl
    *
    100,
    marker="^",
    linewidth=2,
    label="Previous flagged FTEC",
)


plt.plot(
    p_values
    *
    100,
    majority_pl
    *
    100,
    marker="D",
    linewidth=2,
    label="3-round raw majority",
)


plt.plot(
    p_values
    *
    100,
    history_pl
    *
    100,
    marker="P",
    linewidth=2,
    label="3-round history decoder",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Residual Logical Pauli Error Probability (%)"
)

plt.title(
    "Logical Error: Raw Repetition vs Syndrome History"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


LOGICAL_ERROR_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_syndrome_history_logical_error.png",
)


plt.savefig(
    LOGICAL_ERROR_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 3:
# HISTORY GAIN OVER RAW MAJORITY
# ============================================================

history_gain = (
    history_mean
    -
    majority_mean
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
    history_gain
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
    "History Decoder − Majority Accuracy "
    "(percentage points)"
)

plt.title(
    "Classification Value of Temporal Syndrome Information"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


GAIN_FIGURE = os.path.join(
    FIGURE_DIR,
    "cifar10_syndrome_history_gain.png",
)


plt.savefig(
    GAIN_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 4:
# DECODER COSET-PREDICTION ACCURACY
# ============================================================

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
    coset_accuracy
    *
    100,
    marker="o",
    linewidth=2,
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Final Pauli-Coset Prediction Accuracy (%)"
)

plt.title(
    "Syndrome-History Decoder Accuracy"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


DECODER_ACCURACY_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_history_decoder_coset_accuracy.png",
)


plt.savefig(
    DECODER_ACCURACY_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 5:
# TEMPORAL ACTIVITY
# ============================================================

plt.figure(
    figsize=(
        7.8,
        5.5,
    )
)


plt.plot(
    p_values
    *
    100,
    temporal_change
    *
    100,
    marker="o",
    linewidth=2,
    label="Any syndrome change across rounds",
)


plt.plot(
    p_values
    *
    100,
    flag_cycle_rate
    *
    100,
    marker="s",
    linewidth=2,
    label="Any flag across rounds",
)


plt.xlabel(
    "Physical Depolarizing Error Probability (%)"
)

plt.ylabel(
    "Fraction of EC Cycles (%)"
)

plt.title(
    "Temporal Syndrome and Flag Activity"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


TEMPORAL_FIGURE = os.path.join(
    FIGURE_DIR,
    "five_qubit_syndrome_history_activity.png",
)


plt.savefig(
    TEMPORAL_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FILES
# ============================================================

print("\n======================================")
print("FILES SAVED")
print("======================================")

print(
    "Decoder-channel CSV:",
    DECODER_CHANNEL_CSV,
)

print(
    "Detailed CIFAR CSV:",
    DETAILED_CSV,
)

print(
    "Summary CIFAR CSV:",
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
    "History-gain figure:",
    GAIN_FIGURE,
)

print(
    "Decoder-accuracy figure:",
    DECODER_ACCURACY_FIGURE,
)

print(
    "Temporal-activity figure:",
    TEMPORAL_FIGURE,
)


print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    "Three complete flagged stabilizer rounds are "
    "simulated with persistent data errors."
)

print(
    "The raw-majority baseline majority-votes the "
    "three measured syndrome values bit-by-bit."
)

print(
    "The history decoder instead uses temporal syndrome "
    "changes d_t = s_t XOR s_(t-1) together with which "
    "stabilizers flagged during the three-round window."
)

print(
    "The history decoder is a 16-bit -> 64-class "
    "maximum-likelihood lookup calibrated only from "
    "synthetic circuit-noise samples."
)

print(
    "No CIFAR labels are used to train the decoder."
)

print(
    "Calibration and evaluation use independent random "
    "samples."
)

print(
    "If History3 improves over Majority3, that directly "
    "demonstrates the value of time-resolved syndrome "
    "information when data faults can occur between rounds."
)

print(
    "Because three complete rounds add substantial QEC "
    "circuit overhead, History3 is not guaranteed to beat "
    "the previous adaptive single-cycle flagged FTEC."
)

print(
    "That comparison quantifies the tradeoff between "
    "additional temporal information and additional "
    "fault opportunities."
)

print(
    "The next research stage after this experiment is "
    "fault-tolerant logical-gate implementation rather "
    "than another syndrome-measurement toy model."
)
