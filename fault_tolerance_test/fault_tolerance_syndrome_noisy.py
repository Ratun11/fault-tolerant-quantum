from braket.circuits import Circuit
from braket.devices import LocalSimulator
from collections import Counter


# ============================================================
# SETTINGS
# ============================================================

P_SYNDROME = 0.10
SHOTS = 10000

device = LocalSimulator(backend="braket_dm")


# ============================================================
# SOFTWARE RECOVERY
# ============================================================

def correct_data(data_string, syndrome):

    data = list(data_string)

    # Syndrome:
    # 10 -> q0 error
    # 11 -> q1 error
    # 01 -> q2 error

    if syndrome == "10":
        data[0] = "1" if data[0] == "0" else "0"

    elif syndrome == "11":
        data[1] = "1" if data[1] == "0" else "0"

    elif syndrome == "01":
        data[2] = "1" if data[2] == "0" else "0"

    return "".join(data)


# ============================================================
# EXPERIMENT 1
# ONE SYNDROME ROUND
# ============================================================

single = Circuit()


# Logical |0>
single.cnot(0, 1)
single.cnot(0, 2)


# Deliberate data error on q1
#
# 000 -> 010
single.x(1)


# Syndrome extraction
#
# q3 = q0 XOR q1
# q4 = q1 XOR q2
single.cnot(0, 3)
single.cnot(1, 3)

single.cnot(1, 4)
single.cnot(2, 4)


# Random syndrome faults
single.bit_flip(
    3,
    probability=P_SYNDROME
)

single.bit_flip(
    4,
    probability=P_SYNDROME
)


single.measure([0, 1, 2, 3, 4])


single_result = device.run(
    single,
    shots=SHOTS
).result()

single_counts = single_result.measurement_counts


# ============================================================
# ANALYZE SINGLE ROUND
# ============================================================

single_correct_syndrome = 0
single_exact_recovery = 0


for bitstring, count in single_counts.items():

    data = bitstring[0:3]

    syndrome = bitstring[3:5]


    # Correct syndrome should be 11
    if syndrome == "11":
        single_correct_syndrome += count


    corrected = correct_data(
        data,
        syndrome
    )


    # Original logical codeword was 000
    if corrected == "000":
        single_exact_recovery += count


single_syndrome_accuracy = (
    single_correct_syndrome / SHOTS
)

single_recovery_rate = (
    single_exact_recovery / SHOTS
)


# ============================================================
# EXPERIMENT 2
# THREE SYNDROME ROUNDS
# ============================================================

triple = Circuit()


# Logical |0>
triple.cnot(0, 1)
triple.cnot(0, 2)


# Same deliberate error on q1
triple.x(1)


# ============================================================
# ROUND 1
# q3 q4
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
# ROUND 2
# q5 q6
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
# ROUND 3
# q7 q8
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
# ANALYZE THREE ROUNDS
# ============================================================

triple_correct_syndrome = 0
triple_exact_recovery = 0


for bitstring, count in triple_counts.items():

    data = bitstring[0:3]

    round1 = bitstring[3:5]
    round2 = bitstring[5:7]
    round3 = bitstring[7:9]


    # --------------------------------------------------------
    # Majority vote for syndrome bit 1
    # --------------------------------------------------------

    first_bits = [
        int(round1[0]),
        int(round2[0]),
        int(round3[0])
    ]

    s0 = (
        1
        if sum(first_bits) >= 2
        else 0
    )


    # --------------------------------------------------------
    # Majority vote for syndrome bit 2
    # --------------------------------------------------------

    second_bits = [
        int(round1[1]),
        int(round2[1]),
        int(round3[1])
    ]

    s1 = (
        1
        if sum(second_bits) >= 2
        else 0
    )


    decoded_syndrome = f"{s0}{s1}"


    if decoded_syndrome == "11":
        triple_correct_syndrome += count


    corrected = correct_data(
        data,
        decoded_syndrome
    )


    if corrected == "000":
        triple_exact_recovery += count


triple_syndrome_accuracy = (
    triple_correct_syndrome / SHOTS
)

triple_recovery_rate = (
    triple_exact_recovery / SHOTS
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("SETTINGS")
print("======================================")

print(
    f"Syndrome-bit error probability = "
    f"{P_SYNDROME}"
)

print(
    f"Shots = {SHOTS}"
)


print("\n======================================")
print("ONE SYNDROME ROUND")
print("======================================")

print(
    f"Correct syndrome rate = "
    f"{single_syndrome_accuracy:.4f}"
)

print(
    f"Exact recovery rate   = "
    f"{single_recovery_rate:.4f}"
)


print("\n======================================")
print("THREE SYNDROME ROUNDS")
print("======================================")

print(
    f"Correct syndrome rate = "
    f"{triple_syndrome_accuracy:.4f}"
)

print(
    f"Exact recovery rate   = "
    f"{triple_recovery_rate:.4f}"
)


print("\n======================================")
print("IMPROVEMENT")
print("======================================")

print(
    f"Single round failure  = "
    f"{1 - single_recovery_rate:.4f}"
)

print(
    f"Three round failure   = "
    f"{1 - triple_recovery_rate:.4f}"
)