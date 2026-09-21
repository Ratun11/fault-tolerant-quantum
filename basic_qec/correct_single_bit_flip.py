from braket.circuits import Circuit
from braket.devices import LocalSimulator


# ============================================================
# CHOOSE WHERE TO PUT THE ERROR
#
# 0    -> error on q0
# 1    -> error on q1
# 2    -> error on q2
# None -> no error
# ============================================================

ERROR_QUBIT = 2


circuit = Circuit()


# ============================================================
# 1. PREPARE LOGICAL STATE
#
# |0> --H--> |+>
# ============================================================

circuit.h(0)


# ============================================================
# 2. ENCODE
#
# |+> -> (|000> + |111>) / sqrt(2)
# ============================================================

circuit.cnot(0, 1)
circuit.cnot(0, 2)


# ============================================================
# 3. INTRODUCE ONE X ERROR
# ============================================================

if ERROR_QUBIT is not None:
    circuit.x(ERROR_QUBIT)


# ============================================================
# 4. SYNDROME EXTRACTION
#
# q3 = q0 XOR q1
# q4 = q1 XOR q2
# ============================================================

circuit.cnot(0, 3)
circuit.cnot(1, 3)

circuit.cnot(1, 4)
circuit.cnot(2, 4)


# ============================================================
# 5. RECOVERY
# ============================================================


# ------------------------------------------------------------
# Syndrome 10 -> error on q0
#
# We want:
# q3 = 1
# q4 = 0
#
# CCNOT normally activates on 11.
#
# So temporarily flip q4:
#
# 10 -> 11
#
# Apply correction to q0.
#
# Then restore q4:
#
# 11 -> 10
# ------------------------------------------------------------

circuit.x(4)
circuit.ccnot(3, 4, 0)
circuit.x(4)


# ------------------------------------------------------------
# Syndrome 11 -> error on q1
# ------------------------------------------------------------

circuit.ccnot(3, 4, 1)


# ------------------------------------------------------------
# Syndrome 01 -> error on q2
#
# Temporarily:
#
# 01 -> 11
#
# correct q2,
# then restore syndrome.
# ------------------------------------------------------------

circuit.x(3)
circuit.ccnot(3, 4, 2)
circuit.x(3)


# ============================================================
# 6. MEASURE
#
# q0 q1 q2 = corrected data
# q3 q4    = syndrome
# ============================================================

circuit.measure([0, 1, 2, 3, 4])


# ============================================================
# 7. RUN
# ============================================================

print("Error inserted on:", ERROR_QUBIT)

print("\nQuantum circuit:")
print(circuit)


device = LocalSimulator()

task = device.run(
    circuit,
    shots=100
)

result = task.result()


print("\nMeasured qubits:")
print(result.measured_qubits)

print("\nMeasurement counts:")
print(result.measurement_counts)