from braket.circuits import Circuit
from braket.devices import LocalSimulator


# Create circuit
circuit = Circuit()


# Apply Hadamard gate
circuit.h(0)


# Print circuit
print("Quantum circuit:")
print(circuit)


# Local simulator
device = LocalSimulator()


# Run 100 times
task = device.run(circuit, shots=100)


# Get result
result = task.result()


# Print counts
print("\nMeasurement counts:")
print(result.measurement_counts)