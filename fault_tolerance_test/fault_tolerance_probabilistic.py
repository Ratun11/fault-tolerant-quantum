from braket.circuits import Circuit, Noise
from braket.devices import LocalSimulator


# ============================================================
# SETTINGS
# ============================================================

P_DATA = 0.02
P_CNOT = 0.005
P_READOUT = 0.02

SHOTS = 10000

device = LocalSimulator(backend="braket_dm")


# ============================================================
# NOISY CNOT
#
# Apply CNOT, then allow either participating qubit
# to suffer an X error.
# ============================================================

def noisy_cnot(circuit, control, target):

    circuit.cnot(control, target)

    circuit.bit_flip(
        control,
        probability=P_CNOT
    )

    circuit.bit_flip(
        target,
        probability=P_CNOT
    )


# ============================================================
# SOFTWARE CORRECTION
# ============================================================

def correct_data(data_string, syndrome):

    data = list(data_string)

    if syndrome == "10":

        # q0 error
        data[0] = (
            "1" if data[0] == "0"
            else "0"
        )

    elif syndrome == "11":

        # q1 error
        data[1] = (
            "1" if data[1] == "0"
            else "0"
        )

    elif syndrome == "01":

        # q2 error
        data[2] = (
            "1" if data[2] == "0"
            else "0"
        )

    return "".join(data)


# ============================================================
# LOGICAL DECODER
# ============================================================

def logical_value(data):

    return (
        1
        if data.count("1") >= 2
        else 0
    )


# ============================================================
# EXPERIMENT 1
# UNPROTECTED QUBIT
# ============================================================

physical = Circuit()


# Storage/data error
physical.bit_flip(
    0,
    probability=P_DATA
)


# Measure
physical.measure(0)


# Readout error
physical.apply_readout_noise(
    Noise.BitFlip(
        probability=P_READOUT
    )
)


result = device.run(
    physical,
    shots=SHOTS
).result()

physical_counts = result.measurement_counts


physical_error_rate = (
    physical_counts.get("1", 0)
    / SHOTS
)


# ============================================================
# FUNCTION:
# BUILD QEC CIRCUIT
# ============================================================

def build_qec(rounds):

    circuit = Circuit()


    # ========================================================
    # 1. ENCODE |0_L> = |000>
    #
    # Encoding gates are ALSO noisy.
    # ========================================================

    noisy_cnot(
        circuit,
        0,
        1
    )

    noisy_cnot(
        circuit,
        0,
        2
    )


    # ========================================================
    # 2. STORAGE / DATA NOISE
    # ========================================================

    for q in [0, 1, 2]:

        circuit.bit_flip(
            q,
            probability=P_DATA
        )


    # ========================================================
    # 3. SYNDROME EXTRACTION
    # ========================================================

    syndrome_qubits = []

    next_ancilla = 3


    for _ in range(rounds):

        a = next_ancilla
        b = next_ancilla + 1


        # --------------------------------------------
        # syndrome bit 0
        #
        # a = q0 XOR q1
        # --------------------------------------------

        noisy_cnot(
            circuit,
            0,
            a
        )

        noisy_cnot(
            circuit,
            1,
            a
        )


        # --------------------------------------------
        # syndrome bit 1
        #
        # b = q1 XOR q2
        # --------------------------------------------

        noisy_cnot(
            circuit,
            1,
            b
        )

        noisy_cnot(
            circuit,
            2,
            b
        )


        syndrome_qubits.extend(
            [a, b]
        )

        next_ancilla += 2


    # ========================================================
    # 4. MEASURE DATA + ANCILLAS
    # ========================================================

    measured = (
        [0, 1, 2]
        + syndrome_qubits
    )

    circuit.measure(
        measured
    )


    # ========================================================
    # 5. READOUT NOISE
    #
    # Applied just before measurement.
    # ========================================================

    circuit.apply_readout_noise(
        Noise.BitFlip(
            probability=P_READOUT
        ),
        target_qubits=measured
    )


    return circuit


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_qec(counts, rounds):

    logical_errors = 0


    for bitstring, count in counts.items():

        # First 3 bits = data
        data = bitstring[0:3]


        # ====================================================
        # SINGLE ROUND
        # ====================================================

        if rounds == 1:

            syndrome = bitstring[3:5]


        # ====================================================
        # THREE ROUNDS
        # ====================================================

        else:

            syndrome_rounds = []

            index = 3

            for _ in range(rounds):

                syndrome_rounds.append(
                    bitstring[index:index + 2]
                )

                index += 2


            # --------------------------------------------
            # Majority vote for syndrome bit 0
            # --------------------------------------------

            s0_votes = [
                int(s[0])
                for s in syndrome_rounds
            ]

            s0 = (
                1
                if sum(s0_votes) >
                rounds // 2
                else 0
            )


            # --------------------------------------------
            # Majority vote for syndrome bit 1
            # --------------------------------------------

            s1_votes = [
                int(s[1])
                for s in syndrome_rounds
            ]

            s1 = (
                1
                if sum(s1_votes) >
                rounds // 2
                else 0
            )


            syndrome = f"{s0}{s1}"


        # ====================================================
        # SOFTWARE RECOVERY
        # ====================================================

        corrected = correct_data(
            data,
            syndrome
        )


        # ====================================================
        # LOGICAL DECODE
        # ====================================================

        logical = logical_value(
            corrected
        )


        # Original state = logical |0>
        if logical == 1:

            logical_errors += count


    return (
        logical_errors / SHOTS
    )


# ============================================================
# ONE SYNDROME ROUND
# ============================================================

single = build_qec(
    rounds=1
)

single_result = device.run(
    single,
    shots=SHOTS
).result()

single_counts = (
    single_result.measurement_counts
)

single_error_rate = analyze_qec(
    single_counts,
    rounds=1
)


# ============================================================
# THREE SYNDROME ROUNDS
# ============================================================

triple = build_qec(
    rounds=3
)

triple_result = device.run(
    triple,
    shots=SHOTS
).result()

triple_counts = (
    triple_result.measurement_counts
)

triple_error_rate = analyze_qec(
    triple_counts,
    rounds=3
)


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("CIRCUIT-LEVEL NOISE SETTINGS")
print("======================================")

print(
    f"Data error probability    = "
    f"{P_DATA}"
)

print(
    f"CNOT-associated error     = "
    f"{P_CNOT}"
)

print(
    f"Readout error             = "
    f"{P_READOUT}"
)

print(
    f"Shots                     = "
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
    f"{single_error_rate:.4f}"
)


print("\n======================================")
print("THREE-ROUND QEC")
print("======================================")

print(
    f"Logical error rate = "
    f"{triple_error_rate:.4f}"
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
    f"{single_error_rate:.4f}"
)

print(
    f"3-round FT-style   = "
    f"{triple_error_rate:.4f}"
)