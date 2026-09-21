from braket.circuits import Circuit
from braket.devices import LocalSimulator
from collections import Counter


SHOTS = 1000

circuit = Circuit()


# ============================================================
# 1. ENCODE LOGICAL |0>
#
# |0_L> = |000>
# ============================================================

circuit.cnot(0, 1)
circuit.cnot(0, 2)


# ============================================================
# 2. INTRODUCE DATA ERROR
#
# Error on q1
# ============================================================

circuit.x(1)


# ============================================================
# 3. SYNDROME ROUND 1
#
# q3 = q0 XOR q1
# q4 = q1 XOR q2
# ============================================================

circuit.cnot(0, 3)
circuit.cnot(1, 3)

circuit.cnot(1, 4)
circuit.cnot(2, 4)


# ============================================================
# 4. SYNDROME ROUND 2
#
# q5 = q0 XOR q1
# q6 = q1 XOR q2
# ============================================================

circuit.cnot(0, 5)
circuit.cnot(1, 5)

circuit.cnot(1, 6)
circuit.cnot(2, 6)


# ============================================================
# INTENTIONALLY CORRUPT ROUND 2
#
# Flip q6 so syndrome round 2 becomes wrong
# ============================================================

circuit.x(6)


# ============================================================
# 5. SYNDROME ROUND 3
#
# q7 = q0 XOR q1
# q8 = q1 XOR q2
# ============================================================

circuit.cnot(0, 7)
circuit.cnot(1, 7)

circuit.cnot(1, 8)
circuit.cnot(2, 8)


# ============================================================
# 6. MEASURE DATA + ALL SYNDROME QUBITS
# ============================================================

circuit.measure([
    0, 1, 2,
    3, 4,
    5, 6,
    7, 8
])


print("Quantum circuit:")
print(circuit)


# ============================================================
# 7. RUN LOCALLY
# ============================================================

device = LocalSimulator()

result = device.run(
    circuit,
    shots=SHOTS
).result()

counts = result.measurement_counts


print("\nRaw measurement counts:")
print(counts)

# ============================================================
# 8. FAULT-TOLERANT SYNDROME DECODER
# ============================================================

syndrome_counter = Counter()


for bitstring, count in counts.items():

    # Measurement order:
    #
    # q0 q1 q2 q3 q4 q5 q6 q7 q8

    data = bitstring[0:3]

    round1 = bitstring[3:5]
    round2 = bitstring[5:7]
    round3 = bitstring[7:9]

    # --------------------------------------------------------
    # Majority vote for syndrome bit 1
    # --------------------------------------------------------

    s0_values = [
        int(round1[0]),
        int(round2[0]),
        int(round3[0])
    ]

    s0 = 1 if sum(s0_values) >= 2 else 0


    # --------------------------------------------------------
    # Majority vote for syndrome bit 2
    # --------------------------------------------------------

    s1_values = [
        int(round1[1]),
        int(round2[1]),
        int(round3[1])
    ]

    s1 = 1 if sum(s1_values) >= 2 else 0


    final_syndrome = f"{s0}{s1}"

    syndrome_counter[final_syndrome] += count


print("\nDecoded syndrome:")
print(syndrome_counter)

# ============================================================
# 9. SOFTWARE RECOVERY / PAULI-FRAME STYLE CORRECTION
# ============================================================

corrected_counts = Counter()


for bitstring, count in counts.items():

    # --------------------------------------------------------
    # Split measurement result
    #
    # q0 q1 q2 | q3 q4 | q5 q6 | q7 q8
    # --------------------------------------------------------

    data = list(bitstring[0:3])

    round1 = bitstring[3:5]
    round2 = bitstring[5:7]
    round3 = bitstring[7:9]


    # --------------------------------------------------------
    # Majority vote for first syndrome bit
    # --------------------------------------------------------

    s0_values = [
        int(round1[0]),
        int(round2[0]),
        int(round3[0])
    ]

    s0 = 1 if sum(s0_values) >= 2 else 0


    # --------------------------------------------------------
    # Majority vote for second syndrome bit
    # --------------------------------------------------------

    s1_values = [
        int(round1[1]),
        int(round2[1]),
        int(round3[1])
    ]

    s1 = 1 if sum(s1_values) >= 2 else 0


    final_syndrome = f"{s0}{s1}"


    # ========================================================
    # RECOVERY RULE
    #
    # 00 -> no correction
    # 10 -> flip q0
    # 11 -> flip q1
    # 01 -> flip q2
    # ========================================================

    if final_syndrome == "10":

        # flip q0
        data[0] = "1" if data[0] == "0" else "0"


    elif final_syndrome == "11":

        # flip q1
        data[1] = "1" if data[1] == "0" else "0"


    elif final_syndrome == "01":

        # flip q2
        data[2] = "1" if data[2] == "0" else "0"


    corrected_data = "".join(data)

    corrected_counts[corrected_data] += count


print("\nCorrected data:")
print(corrected_counts)