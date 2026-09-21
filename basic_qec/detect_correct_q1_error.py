from braket.circuits import Circuit
from braket.devices import LocalSimulator


circuit = Circuit()


# =====================================================
# 1. PREPARE
# Create |+> on q0
# =====================================================

circuit.h(0)


# =====================================================
# 2. ENCODE
#
# |+>  ->  (|000> + |111>) / sqrt(2)
#
# q0, q1, q2 = data qubits
# =====================================================

circuit.cnot(0, 1)
circuit.cnot(0, 2)


# =====================================================
# 3. INTRODUCE ERROR
#
# Deliberately flip q1
# =====================================================

circuit.x(1)


# =====================================================
# 4. SYNDROME EXTRACTION
#
# q3 = q0 XOR q1
# q4 = q1 XOR q2
# =====================================================

circuit.cnot(0, 3)
circuit.cnot(1, 3)

circuit.cnot(1, 4)
circuit.cnot(2, 4)


# =====================================================
# 5. RECOVERY
#
# Syndrome 11 means:
# q3 = 1 AND q4 = 1
#
# Therefore flip q1
# =====================================================

circuit.ccnot(3, 4, 1)


# =====================================================
# 6. MEASURE EVERYTHING
# =====================================================

circuit.measure([0, 1, 2, 3, 4])


# Print circuit
print("Quantum circuit:")
print(circuit)


# Run locally
device = LocalSimulator()

task = device.run(circuit, shots=100)

result = task.result()


print("\nMeasured qubits:")
print(result.measured_qubits)

print("\nMeasurement counts:")
print(result.measurement_counts)