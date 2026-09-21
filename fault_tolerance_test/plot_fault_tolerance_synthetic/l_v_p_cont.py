import os
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# YOUR MEASURED RESULTS
# ============================================================

p_gate = np.array([
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.050
])

transversal = np.array([
    0.000000,
    0.000000,
    0.000250,
    0.000550,
    0.002800,
    0.015900
])

nontransversal = np.array([
    0.000800,
    0.002050,
    0.004800,
    0.010100,
    0.019450,
    0.053700
])


# ============================================================
# OUTPUT FOLDER
# ============================================================

FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# RATIO CALCULATIONS
# ============================================================

trans_over_p = transversal / p_gate
nontrans_over_p = nontransversal / p_gate

trans_over_p2 = transversal / (p_gate ** 2)
nontrans_over_p2 = nontransversal / (p_gate ** 2)


# ============================================================
# PRINT TABLE
# ============================================================

print("\n======================================")
print("RATIO TABLE")
print("======================================")
print("p\t\tT/p\t\tNT/p\t\tT/p^2\t\tNT/p^2")

for p, tp, ntp, tp2, ntp2 in zip(
    p_gate,
    trans_over_p,
    nontrans_over_p,
    trans_over_p2,
    nontrans_over_p2
):
    print(f"{p:.4f}\t\t{tp:.4f}\t\t{ntp:.4f}\t\t{tp2:.4f}\t\t{ntp2:.4f}")


# ============================================================
# FIGURE 1: p_L / p
# If a curve is roughly flat here, it suggests O(p)
# ============================================================

plt.figure(figsize=(7.5, 5.5))

plt.plot(
    p_gate * 100,
    trans_over_p,
    marker="o",
    linewidth=2,
    label="Transversal"
)

plt.plot(
    p_gate * 100,
    nontrans_over_p,
    marker="s",
    linewidth=2,
    label="Non-transversal"
)

plt.xlabel("Physical CNOT Error Rate (%)", fontsize=12)
plt.ylabel(r"$p_L / p$", fontsize=12)
plt.title(r"Scaling Check: $p_L / p$", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

path1 = os.path.join(FIGURE_DIR, "scaling_pL_over_p.png")
plt.savefig(path1, dpi=300, bbox_inches="tight")
print(f"\nFigure saved to: {path1}")
plt.show()


# ============================================================
# FIGURE 2: p_L / p^2
# If a curve is roughly flat here, it suggests O(p^2)
# ============================================================

plt.figure(figsize=(7.5, 5.5))

plt.plot(
    p_gate * 100,
    trans_over_p2,
    marker="o",
    linewidth=2,
    label="Transversal"
)

plt.plot(
    p_gate * 100,
    nontrans_over_p2,
    marker="s",
    linewidth=2,
    label="Non-transversal"
)

plt.xlabel("Physical CNOT Error Rate (%)", fontsize=12)
plt.ylabel(r"$p_L / p^2$", fontsize=12)
plt.title(r"Scaling Check: $p_L / p^2$", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

path2 = os.path.join(FIGURE_DIR, "scaling_pL_over_p2.png")
plt.savefig(path2, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {path2}")
plt.show()


# ============================================================
# FIGURE 3: original logical error plot on log-log axes
# This often makes scaling behavior easier to see
# ============================================================

plt.figure(figsize=(7.5, 5.5))

plt.loglog(
    p_gate,
    transversal,
    marker="o",
    linewidth=2,
    label="Transversal"
)

plt.loglog(
    p_gate,
    nontransversal,
    marker="s",
    linewidth=2,
    label="Non-transversal"
)

plt.xlabel("Physical CNOT Error Probability", fontsize=12)
plt.ylabel("Logical Error Probability", fontsize=12)
plt.title("Logical Error Scaling (Log-Log)", fontsize=13)
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()

path3 = os.path.join(FIGURE_DIR, "logical_error_scaling_loglog.png")
plt.savefig(path3, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {path3}")
plt.show()