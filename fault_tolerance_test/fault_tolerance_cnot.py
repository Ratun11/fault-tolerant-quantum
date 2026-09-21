from braket.circuits import Circuit
from braket.devices import LocalSimulator
from collections import Counter


SHOTS = 1000

# Turn the artificial fault on/off
INSERT_FAULT = True


circuit = Circuit()


# ============================================================
# 1. PREPARE LOGICAL CONTROL |1_L>
#
# q0 q1 q2 = 111
# ============================================================

circuit.x(0)

circuit.cnot(0, 1)
circuit.cnot(0, 2)


# ============================================================
# 2. PREPARE LOGICAL TARGET |0_L>
#
# q3 q4 q5 = 000
#
# These encoding CNOTs do nothing for |0>,
# but we include them to show the logical block.
# ============================================================

circuit.cnot(3, 4)
circuit.cnot(3, 5)


# ============================================================
# 3. TRANSVERSAL LOGICAL CNOT
#
# Logical control block: q0 q1 q2
# Logical target block:  q3 q4 q5
# ============================================================

# Pair 1
circuit.cnot(0, 3)


# Pair 2
circuit.cnot(1, 4)


# ============================================================
# 4. SIMULATE ONE FAULTY PHYSICAL CNOT
#
# Suppose the physical CNOT between q1 and q4 fails
# and causes an X error on BOTH participating qubits.
#
# This represents ONE faulty physical gate.
# ============================================================

if INSERT_FAULT:

    circuit.x(1)
    circuit.x(4)


# Pair 3
circuit.cnot(2, 5)


# ============================================================
# 5. MEASURE BOTH LOGICAL BLOCKS
# ============================================================

circuit.measure([
    0, 1, 2,
    3, 4, 5
])


print("Quantum circuit:")
print(circuit)


# ============================================================
# 6. RUN
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
# 7. LOGICAL DECODER
#
# Majority vote independently inside each logical block.
# ============================================================

logical_counts = Counter()


for bitstring, count in counts.items():

    control_block = bitstring[0:3]
    target_block = bitstring[3:6]


    # Decode control logical qubit
    control_logical = (
        "1"
        if control_block.count("1") >= 2
        else "0"
    )


    # Decode target logical qubit
    target_logical = (
        "1"
        if target_block.count("1") >= 2
        else "0"
    )


    logical_result = (
        control_logical
        + target_logical
    )


    logical_counts[logical_result] += count


print("\nDecoded logical counts:")
print(logical_counts)