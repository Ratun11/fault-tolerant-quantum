from braket.circuits import Circuit
from braket.devices import LocalSimulator

import matplotlib.pyplot as plt
import os
import csv


# ============================================================
# SETTINGS
# ============================================================

P_GATE_VALUES = [
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050
]

SHOTS = 20000

device = LocalSimulator(
    backend="braket_dm"
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

FIGURE_DIR = "figures"
RESULT_DIR = "results"

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOGICAL DECODER
# ============================================================

def decode_block(block):

    if block.count("1") >= 2:
        return "1"

    return "0"


# ============================================================
# CALCULATE LOGICAL ERROR RATE
#
# Correct logical output should be:
#
# control = 1
# target  = 1
#
# so expected result = "11"
# ============================================================

def calculate_logical_error(counts):

    errors = 0
    total = 0

    for bitstring, count in counts.items():

        control_block = bitstring[0:3]
        target_block = bitstring[3:6]

        control_logical = decode_block(
            control_block
        )

        target_logical = decode_block(
            target_block
        )

        logical_result = (
            control_logical
            + target_logical
        )

        total += count

        if logical_result != "11":
            errors += count

    return errors / total


# ============================================================
# PREPARE BLOCKS
#
# Control:
# |1_L> = |111>
#
# Target:
# |0_L> = |000>
# ============================================================

def prepare_blocks(circuit):

    # logical control = 111
    circuit.x(0)

    circuit.cnot(0, 1)
    circuit.cnot(0, 2)

    # target q3 q4 q5
    # already begins as 000


# ============================================================
# NOISY CNOT
# ============================================================

def noisy_cnot(
    circuit,
    control,
    target,
    p_gate
):

    circuit.cnot(
        control,
        target
    )

    # simplified physical gate-error model
    circuit.bit_flip(
        control,
        probability=p_gate
    )

    circuit.bit_flip(
        target,
        probability=p_gate
    )


# ============================================================
# TRANSVERSAL CIRCUIT
# ============================================================

def build_transversal(p_gate):

    circuit = Circuit()

    prepare_blocks(
        circuit
    )

    # Each physical qubit interacts
    # with only one qubit in other block

    noisy_cnot(
        circuit,
        0,
        3,
        p_gate
    )

    noisy_cnot(
        circuit,
        1,
        4,
        p_gate
    )

    noisy_cnot(
        circuit,
        2,
        5,
        p_gate
    )

    circuit.measure([
        0, 1, 2,
        3, 4, 5
    ])

    return circuit


# ============================================================
# NON-TRANSVERSAL CIRCUIT
#
# q0 is reused as control.
#
# This is intentionally a toy BAD construction
# for demonstrating fault propagation.
# ============================================================

def build_nontransversal(p_gate):

    circuit = Circuit()

    prepare_blocks(
        circuit
    )

    noisy_cnot(
        circuit,
        0,
        3,
        p_gate
    )

    noisy_cnot(
        circuit,
        0,
        4,
        p_gate
    )

    noisy_cnot(
        circuit,
        0,
        5,
        p_gate
    )

    circuit.measure([
        0, 1, 2,
        3, 4, 5
    ])

    return circuit


# ============================================================
# STORAGE FOR RESULTS
# ============================================================

transversal_results = []
nontransversal_results = []


# ============================================================
# RUN SWEEP
# ============================================================

print("\n======================================")
print("FAULT-TOLERANT CNOT ERROR SWEEP")
print("======================================")

print(f"Shots per experiment = {SHOTS}")


for p_gate in P_GATE_VALUES:

    print("\n--------------------------------------")

    print(
        f"Physical CNOT error probability = "
        f"{p_gate}"
    )


    # ========================================================
    # TRANSVERSAL
    # ========================================================

    transversal = build_transversal(
        p_gate
    )

    transversal_result = device.run(
        transversal,
        shots=SHOTS
    ).result()

    transversal_counts = (
        transversal_result.measurement_counts
    )

    transversal_error = (
        calculate_logical_error(
            transversal_counts
        )
    )


    # ========================================================
    # NON-TRANSVERSAL
    # ========================================================

    nontransversal = build_nontransversal(
        p_gate
    )

    nontransversal_result = device.run(
        nontransversal,
        shots=SHOTS
    ).result()

    nontransversal_counts = (
        nontransversal_result.measurement_counts
    )

    nontransversal_error = (
        calculate_logical_error(
            nontransversal_counts
        )
    )


    transversal_results.append(
        transversal_error
    )

    nontransversal_results.append(
        nontransversal_error
    )


    print(
        f"Transversal logical error     = "
        f"{transversal_error:.6f}"
    )

    print(
        f"Non-transversal logical error = "
        f"{nontransversal_error:.6f}"
    )


# ============================================================
# PRINT TABLE
# ============================================================

print("\n\n======================================")
print("FINAL RESULTS")
print("======================================")

print(
    "P_GATE\t\tTransversal\tNon-transversal"
)

for p, ft, nft in zip(
    P_GATE_VALUES,
    transversal_results,
    nontransversal_results
):

    print(
        f"{p:.4f}\t\t"
        f"{ft:.6f}\t\t"
        f"{nft:.6f}"
    )


# ============================================================
# SAVE RESULTS TO CSV
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "logical_cnot_error_sweep.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "physical_cnot_error",
        "transversal_logical_error",
        "nontransversal_logical_error"
    ])

    for p, ft, nft in zip(
        P_GATE_VALUES,
        transversal_results,
        nontransversal_results
    ):

        writer.writerow([
            p,
            ft,
            nft
        ])


print(
    f"\nResults saved to: {csv_path}"
)


# ============================================================
# CREATE FIGURE
# ============================================================

physical_percent = [
    p * 100
    for p in P_GATE_VALUES
]

transversal_percent = [
    p * 100
    for p in transversal_results
]

nontransversal_percent = [
    p * 100
    for p in nontransversal_results
]


plt.figure(
    figsize=(7.5, 5.5)
)


plt.plot(
    physical_percent,
    transversal_percent,
    marker="o",
    linewidth=2,
    label="Transversal"
)


plt.plot(
    physical_percent,
    nontransversal_percent,
    marker="s",
    linewidth=2,
    label="Non-transversal"
)


plt.xlabel(
    "Physical CNOT Error Rate (%)",
    fontsize=12
)

plt.ylabel(
    "Logical Error Rate (%)",
    fontsize=12
)

plt.title(
    "Logical Error Rate vs Physical CNOT Error Rate",
    fontsize=13
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE PNG
# ============================================================

figure_path = os.path.join(
    FIGURE_DIR,
    "logical_error_vs_gate_error.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"Figure saved to: {figure_path}"
)

plt.show()