from braket.circuits import Circuit
from braket.devices import LocalSimulator


# ---------------------------
# 1. Create a quantum circuit
# ---------------------------
circuit = Circuit()


# ---------------------------
# 2. Apply an X gate to qubit 0
# ---------------------------
circuit.x(0)


# ---------------------------
# 3. Print the circuit
# ---------------------------
print("Quantum circuit:")
print(circuit)


# ---------------------------
# 4. Create local simulator
# ---------------------------
device = LocalSimulator()


# ---------------------------
# 5. Run the circuit 100 times
# ---------------------------
task = device.run(circuit, shots=100)


# ---------------------------
# 6. Get result
# ---------------------------
result = task.result()


# ---------------------------
# 7. Print measurement result
# ---------------------------
print("\nMeasurement counts:")
print(result.measurement_counts)
