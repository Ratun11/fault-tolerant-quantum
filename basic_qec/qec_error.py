from braket.circuits import Circuit
from braket.devices import LocalSimulator


# ============================================================
# SETTINGS
# ============================================================

P_ERROR = 0.10
SHOTS = 10000


# ============================================================
# NOISY LOCAL SIMULATOR
# ============================================================

device = LocalSimulator(backend="braket_dm")


# ============================================================
# EXPERIMENT 1:
# ONE UNPROTECTED PHYSICAL QUBIT
# ============================================================

physical = Circuit()

# Initial state is |0>
# Apply bit-flip noise with probability p
physical.bit_flip(0, probability=P_ERROR)

physical.measure(0)


physical_result = device.run(
    physical,
    shots=SHOTS
).result()

physical_counts = physical_result.measurement_counts


print("======================================")
print("UNPROTECTED PHYSICAL QUBIT")
print("======================================")

print("\nCircuit:")
print(physical)

print("\nCounts:")
print(physical_counts)


physical_errors = physical_counts.get("1", 0)

physical_error_rate = physical_errors / SHOTS


print("\nPhysical error rate:")
print(physical_error_rate)


# ============================================================
# EXPERIMENT 2:
# 3-QUBIT BIT-FLIP ERROR CORRECTION
# ============================================================

qec = Circuit()


# ------------------------------------------------------------
# 1. ENCODE LOGICAL |0>
#
# |0_L> = |000>
# ------------------------------------------------------------

qec.cnot(0, 1)
qec.cnot(0, 2)


# ------------------------------------------------------------
# 2. APPLY INDEPENDENT BIT-FLIP NOISE
#
# Every physical data qubit has probability p
# of suffering an X error.
# ------------------------------------------------------------

qec.bit_flip(0, probability=P_ERROR)
qec.bit_flip(1, probability=P_ERROR)
qec.bit_flip(2, probability=P_ERROR)


# ------------------------------------------------------------
# 3. SYNDROME EXTRACTION
#
# q3 = q0 XOR q1
# q4 = q1 XOR q2
# ------------------------------------------------------------

qec.cnot(0, 3)
qec.cnot(1, 3)

qec.cnot(1, 4)
qec.cnot(2, 4)


# ------------------------------------------------------------
# 4. RECOVERY
#
# syndrome 10 -> correct q0
# ------------------------------------------------------------

qec.x(4)
qec.ccnot(3, 4, 0)
qec.x(4)


# syndrome 11 -> correct q1

qec.ccnot(3, 4, 1)


# syndrome 01 -> correct q2

qec.x(3)
qec.ccnot(3, 4, 2)
qec.x(3)


# ------------------------------------------------------------
# 5. MEASURE DATA QUBITS
# ------------------------------------------------------------

qec.measure([0, 1, 2])


qec_result = device.run(
    qec,
    shots=SHOTS
).result()

qec_counts = qec_result.measurement_counts


print("\n\n======================================")
print("3-QUBIT ERROR-CORRECTED LOGICAL QUBIT")
print("======================================")

print("\nCircuit:")
print(qec)

print("\nCounts:")
print(qec_counts)


# Logical |0> should finish as 000.
#
# If >= 2 physical qubits flip,
# the repetition code can fail and produce logical 111.

logical_errors = qec_counts.get("111", 0)

logical_error_rate = logical_errors / SHOTS


print("\nLogical error rate:")
print(logical_error_rate)


# ============================================================
# COMPARISON
# ============================================================

print("\n\n======================================")
print("COMPARISON")
print("======================================")

print(f"Physical error probability p = {P_ERROR}")

print(
    f"Observed unprotected error rate = "
    f"{physical_error_rate:.4f}"
)

print(
    f"Observed logical error rate     = "
    f"{logical_error_rate:.4f}"
)


# ============================================================
# THEORETICAL LOGICAL ERROR RATE
#
# Code fails if:
#
# exactly 2 qubits flip
# OR
# all 3 qubits flip
#
# P_L = 3 p^2 (1-p) + p^3
# ============================================================

theoretical_logical_error = (
    3 * (P_ERROR ** 2) * (1 - P_ERROR)
    + P_ERROR ** 3
)

print(
    f"Theoretical logical error rate  = "
    f"{theoretical_logical_error:.4f}"
)