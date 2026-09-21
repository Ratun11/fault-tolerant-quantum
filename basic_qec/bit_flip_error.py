from braket.circuits import Circuit
from braket.devices import LocalSimulator


# Create circuit
circuit = Circuit()


# ---------------------------------
# Step 1: Prepare |+> on qubit 0
# ---------------------------------
circuit.h(0)


# ---------------------------------
# Step 2: Encode into 3 qubits
# |+> -> (|000> + |111>) / sqrt(2)
# ---------------------------------
circuit.cnot(0, 1)
circuit.cnot(0, 2)


# ---------------------------------
# Step 3: Introduce an error
# Flip physical qubit q1
# ---------------------------------
circuit.x(1)


# Print circuit
print("Quantum circuit:")
print(circuit)


# Local simulator
device = LocalSimulator()


# Run circuit
task = device.run(circuit, shots=100)

result = task.result()


# Print result
print("\nMeasurement counts:")
print(result.measurement_counts)