from braket.circuits import Circuit
from braket.devices import LocalSimulator


circuit = Circuit()


# ---------------------------------
# 1. Prepare |+> on q0
# ---------------------------------
circuit.h(0)


# ---------------------------------
# 2. Encode logical qubit
# |+> -> (|000> + |111>) / sqrt(2)
# ---------------------------------
circuit.cnot(0, 1)
circuit.cnot(0, 2)


# ---------------------------------
# 3. Introduce X error on q1
# ---------------------------------
circuit.x(0)


# ---------------------------------
# 4. Syndrome check 1
# q3 = q0 XOR q1
# ---------------------------------
circuit.cnot(0, 3)
circuit.cnot(1, 3)


# ---------------------------------
# 5. Syndrome check 2
# q4 = q1 XOR q2
# ---------------------------------
circuit.cnot(1, 4)
circuit.cnot(2, 4)


# ---------------------------------
# 6. Measure ONLY syndrome qubits
# ---------------------------------
circuit.measure([3, 4])


print("Quantum circuit:")
print(circuit)


device = LocalSimulator()

task = device.run(circuit, shots=100)

result = task.result()


print("\nMeasured qubits:")
print(result.measured_qubits)

print("\nSyndrome counts:")
print(result.measurement_counts)