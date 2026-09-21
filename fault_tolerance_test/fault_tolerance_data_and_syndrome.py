from braket.circuits import Circuit
from braket.devices import LocalSimulator


# ============================================================
# SETTINGS
# ============================================================

P_DATA = 0.05
P_SYNDROME = 0.10
SHOTS = 10000

device = LocalSimulator(backend="braket_dm")


# ============================================================
# HELPER: APPLY SOFTWARE CORRECTION
# ============================================================

def correct_data(data_string, syndrome):

    data = list(data_string)

    if syndrome == "10":
        # error on q0
        data[0] = "1" if data[0] == "0" else "0"

    elif syndrome == "11":
        # error on q1
        data[1] = "1" if data[1] == "0" else "0"

    elif syndrome == "01":
        # error on q2
        data[2] = "1" if data[2] == "0" else "0"

    return "".join(data)


# ============================================================
# HELPER: DECODE LOGICAL VALUE BY MAJORITY
# ============================================================

def logical_value(data_string):

    ones = data_string.count("1")

    if ones >= 2:
        return 1

    return 0


# ============================================================
# EXPERIMENT 1
# UNPROTECTED PHYSICAL QUBIT
# ============================================================

physical = Circuit()

# Start in |0>
physical.bit_flip(
    0,
    probability=P_DATA
)

physical.measure(0)


physical_result = device.run(
    physical,
    shots=SHOTS
).result()

physical_counts = physical_result.measurement_counts

physical_errors = physical_counts.get("1", 0)

physical_error_rate = (
    physical_errors / SHOTS
)


# ============================================================
# EXPERIMENT 2
# ONE SYNDROME ROUND
# ============================================================

single = Circuit()


# ------------------------------------------------------------
# Encode logical |0>
#
# |0_L> = |000>
# ------------------------------------------------------------

single.cnot(0, 1)
single.cnot(0, 2)


# ------------------------------------------------------------
# RANDOM DATA ERRORS
#
# Each data qubit independently has probability P_DATA
# of suffering an X error.
# ------------------------------------------------------------

single.bit_flip(
    0,
    probability=P_DATA
)

single.bit_flip(
    1,
    probability=P_DATA
)

single.bit_flip(
    2,
    probability=P_DATA
)


# ------------------------------------------------------------
# Syndrome extraction
#
# q3 = q0 XOR q1
# q4 = q1 XOR q2
# ------------------------------------------------------------

single.cnot(0, 3)
single.cnot(1, 3)

single.cnot(1, 4)
single.cnot(2, 4)


# ------------------------------------------------------------
# SYNDROME NOISE
# ------------------------------------------------------------

single.bit_flip(
    3,
    probability=P_SYNDROME
)

single.bit_flip(
    4,
    probability=P_SYNDROME
)


single.measure([
    0, 1, 2,
    3, 4
])


single_result = device.run(
    single,
    shots=SHOTS
).result()

single_counts = single_result.measurement_counts


# ============================================================
# ANALYZE SINGLE-ROUND QEC
# ============================================================

single_logical_errors = 0


for bitstring, count in single_counts.items():

    data = bitstring[0:3]

    syndrome = bitstring[3:5]

    corrected_data = correct_data(
        data,
        syndrome
    )

    logical = logical_value(
        corrected_data
    )

    # Original logical state was |0>
    if logical == 1:
        single_logical_errors += count


single_logical_error_rate = (
    single_logical_errors / SHOTS
)


# ============================================================
# EXPERIMENT 3
# THREE SYNDROME ROUNDS
# ============================================================

triple = Circuit()


# ------------------------------------------------------------
# Encode |0_L>
# ------------------------------------------------------------

triple.cnot(0, 1)
triple.cnot(0, 2)


# ------------------------------------------------------------
# DATA NOISE
# ------------------------------------------------------------

triple.bit_flip(
    0,
    probability=P_DATA
)

triple.bit_flip(
    1,
    probability=P_DATA
)

triple.bit_flip(
    2,
    probability=P_DATA
)


# ============================================================
# SYNDROME ROUND 1
#
# q3, q4
# ============================================================

triple.cnot(0, 3)
triple.cnot(1, 3)

triple.cnot(1, 4)
triple.cnot(2, 4)

triple.bit_flip(
    3,
    probability=P_SYNDROME
)

triple.bit_flip(
    4,
    probability=P_SYNDROME
)


# ============================================================
# SYNDROME ROUND 2
#
# q5, q6
# ============================================================

triple.cnot(0, 5)
triple.cnot(1, 5)

triple.cnot(1, 6)
triple.cnot(2, 6)

triple.bit_flip(
    5,
    probability=P_SYNDROME
)

triple.bit_flip(
    6,
    probability=P_SYNDROME
)


# ============================================================
# SYNDROME ROUND 3
#
# q7, q8
# ============================================================

triple.cnot(0, 7)
triple.cnot(1, 7)

triple.cnot(1, 8)
triple.cnot(2, 8)

triple.bit_flip(
    7,
    probability=P_SYNDROME
)

triple.bit_flip(
    8,
    probability=P_SYNDROME
)


triple.measure([
    0, 1, 2,
    3, 4,
    5, 6,
    7, 8
])


triple_result = device.run(
    triple,
    shots=SHOTS
).result()

triple_counts = triple_result.measurement_counts


# ============================================================
# ANALYZE THREE-ROUND QEC
# ============================================================

triple_logical_errors = 0


for bitstring, count in triple_counts.items():

    data = bitstring[0:3]

    round1 = bitstring[3:5]
    round2 = bitstring[5:7]
    round3 = bitstring[7:9]


    # --------------------------------------------------------
    # Majority vote:
    # syndrome bit 0
    # --------------------------------------------------------

    s0_votes = [
        int(round1[0]),
        int(round2[0]),
        int(round3[0])
    ]

    s0 = (
        1
        if sum(s0_votes) >= 2
        else 0
    )


    # --------------------------------------------------------
    # Majority vote:
    # syndrome bit 1
    # --------------------------------------------------------

    s1_votes = [
        int(round1[1]),
        int(round2[1]),
        int(round3[1])
    ]

    s1 = (
        1
        if sum(s1_votes) >= 2
        else 0
    )


    decoded_syndrome = (
        f"{s0}{s1}"
    )


    # --------------------------------------------------------
    # Software recovery
    # --------------------------------------------------------

    corrected_data = correct_data(
        data,
        decoded_syndrome
    )


    # --------------------------------------------------------
    # Decode logical value
    # --------------------------------------------------------

    logical = logical_value(
        corrected_data
    )


    if logical == 1:
        triple_logical_errors += count


triple_logical_error_rate = (
    triple_logical_errors / SHOTS
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("SETTINGS")
print("======================================")

print(
    f"Data error probability     = "
    f"{P_DATA}"
)

print(
    f"Syndrome error probability = "
    f"{P_SYNDROME}"
)

print(
    f"Shots                      = "
    f"{SHOTS}"
)


print("\n======================================")
print("UNPROTECTED")
print("======================================")

print(
    f"Physical error rate = "
    f"{physical_error_rate:.4f}"
)


print("\n======================================")
print("ONE-ROUND QEC")
print("======================================")

print(
    f"Logical error rate = "
    f"{single_logical_error_rate:.4f}"
)


print("\n======================================")
print("THREE-ROUND QEC")
print("======================================")

print(
    f"Logical error rate = "
    f"{triple_logical_error_rate:.4f}"
)


print("\n======================================")
print("SUMMARY")
print("======================================")

print(
    f"Unprotected        = "
    f"{physical_error_rate:.4f}"
)

print(
    f"1-round QEC        = "
    f"{single_logical_error_rate:.4f}"
)

print(
    f"3-round FT-style   = "
    f"{triple_logical_error_rate:.4f}"
)