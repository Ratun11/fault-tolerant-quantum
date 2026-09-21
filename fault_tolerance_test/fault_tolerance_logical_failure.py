from braket.circuits import Circuit
from braket.devices import LocalSimulator


# ============================================================
# SETTINGS
# ============================================================

P_GATE = 0.01
SHOTS = 10000

device = LocalSimulator(backend="braket_dm")


# ============================================================
# HELPER:
# NOISY PHYSICAL CNOT
#
# After each CNOT, both participating physical qubits
# independently have probability P_GATE of suffering X.
# ============================================================

def noisy_cnot(circuit, control, target):

    circuit.cnot(control, target)

    circuit.bit_flip(
        control,
        probability=P_GATE
    )

    circuit.bit_flip(
        target,
        probability=P_GATE
    )


# ============================================================
# LOGICAL DECODER
# ============================================================

def decode_logical_block(block):

    return (
        "1"
        if block.count("1") >= 2
        else "0"
    )


# ============================================================
# ANALYZE A 6-QUBIT RESULT
#
# Expected correct logical result:
#
# |1_L>|0_L>
#       |
#       CNOT
#       v
# |1_L>|1_L>
#
# Therefore expected logical output = "11"
# ============================================================

def logical_error_rate(counts):

    errors = 0
    total = 0

    for bitstring, count in counts.items():

        total += count

        control_block = bitstring[0:3]
        target_block = bitstring[3:6]

        control_logical = decode_logical_block(
            control_block
        )

        target_logical = decode_logical_block(
            target_block
        )

        logical_result = (
            control_logical
            + target_logical
        )

        if logical_result != "11":
            errors += count

    return errors / total


# ============================================================
# PREPARE BOTH LOGICAL BLOCKS
#
# Control = |1_L> = |111>
# Target  = |0_L> = |000>
#
# We keep preparation perfect here so that we isolate
# the effect of the logical-CNOT construction itself.
# ============================================================

def prepare_blocks(circuit):

    # Control block
    circuit.x(0)
    circuit.cnot(0, 1)
    circuit.cnot(0, 2)

    # Target block q3 q4 q5 already = 000


# ============================================================
# EXPERIMENT 1:
# TRANSVERSAL LOGICAL CNOT
# ============================================================

transversal = Circuit()

prepare_blocks(transversal)


# Each physical control interacts with only one target
noisy_cnot(transversal, 0, 3)
noisy_cnot(transversal, 1, 4)
noisy_cnot(transversal, 2, 5)


transversal.measure([
    0, 1, 2,
    3, 4, 5
])


transversal_result = device.run(
    transversal,
    shots=SHOTS
).result()

transversal_counts = (
    transversal_result.measurement_counts
)

transversal_error = logical_error_rate(
    transversal_counts
)


# ============================================================
# EXPERIMENT 2:
# NON-TRANSVERSAL / REUSED CONTROL
# ============================================================

bad = Circuit()

prepare_blocks(bad)


# Same physical control q0 is reused
noisy_cnot(bad, 0, 3)
noisy_cnot(bad, 0, 4)
noisy_cnot(bad, 0, 5)


bad.measure([
    0, 1, 2,
    3, 4, 5
])


bad_result = device.run(
    bad,
    shots=SHOTS
).result()

bad_counts = bad_result.measurement_counts

bad_error = logical_error_rate(
    bad_counts
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("SETTINGS")
print("======================================")

print(
    f"Physical CNOT fault probability = "
    f"{P_GATE}"
)

print(
    f"Shots                           = "
    f"{SHOTS}"
)


print("\n======================================")
print("TRANSVERSAL LOGICAL CNOT")
print("======================================")

print(
    f"Logical error rate = "
    f"{transversal_error:.4f}"
)


print("\n======================================")
print("NON-TRANSVERSAL LOGICAL CNOT")
print("======================================")

print(
    f"Logical error rate = "
    f"{bad_error:.4f}"
)


print("\n======================================")
print("COMPARISON")
print("======================================")

print(
    f"Transversal     = "
    f"{transversal_error:.4f}"
)

print(
    f"Non-transversal = "
    f"{bad_error:.4f}"
)