import os
import matplotlib.pyplot as plt


# ============================================================
# RESULTS FROM YOUR EXPERIMENT
# ============================================================

transversal_error = 0.0004
nontransversal_error = 0.0100


# ============================================================
# CREATE FIGURE FOLDER
# ============================================================

FIGURE_DIR = "figures"

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# DATA
# ============================================================

methods = [
    "Transversal",
    "Non-transversal"
]

error_rates = [
    transversal_error,
    nontransversal_error
]

# Convert to percentage
error_percent = [
    value * 100
    for value in error_rates
]


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(7, 5))

bars = plt.bar(
    methods,
    error_percent
)


# Add value labels
for bar, value in zip(bars, error_percent):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )


plt.xlabel(
    "Logical CNOT Construction",
    fontsize=12
)

plt.ylabel(
    "Logical Error Rate (%)",
    fontsize=12
)

plt.title(
    "Transversal vs Non-Transversal Logical CNOT",
    fontsize=13
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE PNG
# ============================================================

figure_path = os.path.join(
    FIGURE_DIR,
    "transversal_vs_nontransversal.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved to: {figure_path}")

plt.show()