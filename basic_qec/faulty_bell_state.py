from braket.circuits import Circuit
from braket.devices import LocalSimulator


# Create circuit
circuit = Circuit()

# Step 1: Put qubit 0 into superposition
circuit.h(0)

# Step 2: Entangle qubit 1 with qubit 0
circuit.cnot(0, 1)


# Print circuit
print("Quantum circuit:")
print(circuit)


# Run locally
device = LocalSimulator()

task = device.run(circuit, shots=100)

result = task.result()

print("\nMeasurement counts:")
print(result.measurement_counts)