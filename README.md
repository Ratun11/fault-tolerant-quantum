# Fault-Tolerant Quantum Classification on CIFAR-10

This repository contains a progressive simulation study of **quantum error correction (QEC)** and **fault-tolerant quantum computation (FTQC)** for a hybrid quantum-classical image-classification pipeline based on **CIFAR-10**.

The project starts from elementary quantum error-correction experiments and gradually introduces increasingly realistic sources of failure, including noisy syndrome measurements, circuit-level faults, fault propagation, general Pauli noise, the `[[5,1,3]]` perfect code, flagged fault-tolerant syndrome extraction, temporal syndrome decoding, and logical-gate overhead.

The final application is an **8-logical-qubit hybrid quantum-classical classifier** using pretrained ResNet18 features and a variational quantum circuit.

The ideal hybrid classifier reaches:

- **CIFAR-10 test accuracy:** `91.43%`
- **Macro-F1:** `91.46%`

The main purpose of this repository is not to claim quantum advantage, but to study how progressively more realistic fault-tolerance assumptions affect an application-level quantum machine-learning workload.

---

## Overview

A common QEC simulation assumes that error correction itself is perfect. Under that assumption, quantum codes can appear extremely effective.

This project investigates what happens when that assumption is gradually removed.

The experimental progression is:

```text
Basic QEC
    ↓
Three-qubit repetition code
    ↓
Noisy syndrome measurements
    ↓
Fault propagation and transversal operations
    ↓
Circuit-level noisy QEC
    ↓
CIFAR-10 hybrid quantum classifier
    ↓
General Pauli / depolarizing noise
    ↓
[[5,1,3]] perfect quantum code
    ↓
Noisy stabilizer extraction
    ↓
Flagged fault-tolerant error correction
    ↓
Syndrome-history decoding
    ↓
Gate-aware logical fault model
    ↓
Final end-to-end FT CIFAR-10 analysis
```

The experiments reveal that strong block-level QEC performance does not automatically translate into application-level fault tolerance. The accumulated overhead of syndrome extraction, recovery, repeated correction, and logical operations can dominate the benefit of error suppression.

---

# Main Research Questions

This repository investigates the following questions:

1. How effectively does simple QEC suppress physical errors in small quantum codes?
2. How does noisy syndrome extraction affect QEC performance?
3. Why is fault propagation important in fault-tolerant quantum circuit design?
4. When does repeated syndrome measurement improve reliability?
5. How does a general Pauli channel affect a quantum machine-learning circuit?
6. How well does the `[[5,1,3]]` perfect code protect an 8-qubit classifier?
7. Can flagged syndrome extraction reduce correlated hook errors?
8. Does temporal syndrome history outperform simple majority voting?
9. How does logical-gate overhead affect end-to-end fault-tolerant inference?
10. Under what conditions does fault-tolerant protection become beneficial at the application level?

---

# Hybrid CIFAR-10 Classifier

The application-level benchmark is a hybrid quantum-classical image classifier.

The pipeline is:

```text
CIFAR-10 image
      ↓
Pretrained ResNet18
      ↓
Learned 32-dimensional feature representation
      ↓
8-qubit variational quantum circuit
      ↓
8 expectation values <Z>
      ↓
Classical neural-network head
      ↓
10 CIFAR-10 classes
```

The quantum circuit uses **4 data-reuploading layers**.

Each layer contains:

```text
8 data-encoding RY gates
8 trainable RY gates
8 trainable RZ gates
8 CNOT gates in a ring
```

Therefore one inference contains:

```text
32 data RY gates
32 trainable RY gates
32 trainable RZ gates
32 logical CNOT gates
```

or:

```text
96 arbitrary single-qubit rotations
32 logical CNOTs
```

Counting both qubit participants of each CNOT gives approximately:

```text
160 modeled logical fault locations
```

in the uniform logical-noise model.

---

# Ideal Classifier Performance

The trained hybrid model achieves:

| Metric | Value |
|---|---:|
| CIFAR-10 test accuracy | **91.43%** |
| Macro-F1 | **91.46%** |
| Logical qubits | 8 |
| Input feature dimension | 32 |
| Data-reuploading layers | 4 |

The quantum circuit is evaluated using a custom PyTorch statevector simulator running on GPU.

The simulator represents:

```text
8 qubits → 2^8 = 256 complex amplitudes
```

per sample.

The implementation was validated against the trained quantum model in the zero-noise case.

---

# Stage 1: Three-Qubit Repetition Code

The first QEC experiments use the standard three-qubit repetition code for bit-flip protection.

Logical encoding:

```text
|0L> = |000>
|1L> = |111>
```

The code corrects one physical `X` error.

The syndrome mapping used in the experiments is:

| Syndrome | Error |
|---|---|
| `00` | No error |
| `10` | Error on qubit 0 |
| `11` | Error on qubit 1 |
| `01` | Error on qubit 2 |

For independent physical bit-flip probability `p`, the ideal logical error probability is

```text
pL = 3p^2 - 2p^3
```

For example, at:

```text
p = 0.10
```

the simulation obtained approximately:

```text
Unprotected error ≈ 9.96%
QEC logical error ≈ 2.83%
Theory ≈ 2.80%
```

This validates the basic QEC implementation.

---

# Stage 2: Noisy Syndrome Measurements

Syndrome measurements were then made noisy.

Example configuration:

```text
data error probability      = 0.05
syndrome error probability  = 0.10
```

Repeated syndrome extraction was evaluated.

Observed error rates:

```text
Unprotected ≈ 4.72%
1-round QEC ≈ 2.32%
3-round QEC ≈ 1.11%
```

Under this simplified model, repeated syndrome measurements and majority voting improve reliability.

This result changes later when the extraction circuitry itself becomes noisy.

---

# Stage 3: Fault Propagation and Transversal Gates

The next experiment compares a transversal logical CNOT with a deliberately non-transversal construction.

The measured logical error scaling was approximately:

```text
Transversal circuit:
pL ~ O(p^1.86)

Non-transversal circuit:
pL ~ O(p^1.05)
```

The near-quadratic behavior of the transversal implementation demonstrates the core FT principle that a single physical fault should not be allowed to spread into an uncorrectable multi-qubit error.

Representative values:

| Physical error `p` | Transversal | Non-transversal |
|---:|---:|---:|
| 0.001 | 0.00000 | 0.00080 |
| 0.002 | 0.00000 | 0.00205 |
| 0.005 | 0.00025 | 0.00480 |
| 0.010 | 0.00055 | 0.01010 |
| 0.020 | 0.00280 | 0.01945 |
| 0.050 | 0.01590 | 0.05370 |

---

# Stage 4: CIFAR-10 Noise Sensitivity

The trained 8-qubit quantum classifier was evaluated under quantum noise.

Without QEC, accuracy decreases rapidly as errors accumulate across the quantum circuit.

Representative results under earlier bit-flip-style experiments include:

| Physical error `p` | Unprotected accuracy |
|---:|---:|
| 0.000 | 91.43% |
| 0.001 | 91.29% |
| 0.002 | 91.06% |
| 0.005 | 88.87% |
| 0.010 | 70.75% |
| 0.020 | 37.57% |
| 0.050 | 19.75% |

This demonstrates why application-level QEC is needed: even relatively small per-location error probabilities accumulate over many circuit operations.

---

# Stage 5: Ideal Repetition-Code Protection

An ideal repetition-code logical-error proxy was applied to the classifier using:

```text
pL = 3p^2 - 2p^3
```

Representative results:

| `p` | Unprotected | Ideal repetition QEC |
|---:|---:|---:|
| 0.000 | 91.43% | 91.43% |
| 0.001 | 91.29% | 91.43% |
| 0.002 | 91.06% | 91.43% |
| 0.005 | 88.87% | 91.43% |
| 0.010 | 70.75% | 91.41% |
| 0.020 | 37.57% | 91.27% |
| 0.050 | 19.75% | 82.53% |

This experiment represents an optimistic ideal-QEC upper bound.

---

# Stage 6: Circuit-Level Repetition-Code QEC

The correction circuit was then assigned explicit error rates.

Example parameters:

```text
p_CNOT      = 0.005
p_measure   = 0.010
p_recovery  = 0.005
```

Unlike ideal QEC, circuit-level QEC produces an error floor even when the externally applied data error is zero.

For example:

```text
p_data = 0
```

gave approximately:

```text
1-round QEC accuracy ≈ 88.40%
3-round QEC accuracy ≈ 85.91%
```

This demonstrates that extra syndrome rounds can become harmful when they introduce additional faulty operations.

A break-even heatmap was also generated over combinations of:

```text
data error
CNOT error
measurement error
```

showing that the preferred number of syndrome rounds depends strongly on the physical error regime.

---

# Stage 7: General Pauli Noise

The simplified bit-flip model was replaced with the single-qubit depolarizing channel

```text
E(ρ) =
(1-p)ρ
+
p/3 (XρX + YρY + ZρZ)
```

This introduces all three nontrivial Pauli errors:

```text
X
Y
Z
```

and provides a more general error model for the subsequent five-qubit-code experiments.

---

# Stage 8: The [[5,1,3]] Perfect Code

The `[[5,1,3]]` perfect code encodes:

```text
1 logical qubit
```

into:

```text
5 physical qubits
```

and corrects an arbitrary single-qubit Pauli error.

The stabilizer generators used are:

```text
X Z Z X I
I X Z Z X
X I X Z Z
Z X I X Z
```

with logical operators:

```text
XL = X X X X X
ZL = Z Z Z Z Z
```

For the 8-logical-qubit classifier, full encoding corresponds conceptually to:

```text
8 × 5 = 40 physical data qubits
```

before syndrome ancillas are included.

Rather than simulating a full 40-qubit statevector, the implementation retains the 8-qubit logical statevector and evaluates encoded blocks through stabilizer and Pauli-frame calculations.

---

# Exact Ideal Five-Qubit Logical Channel

All possible physical Pauli patterns were enumerated:

```text
4^5 = 1024
```

For each error pattern:

1. The stabilizer syndrome was computed.
2. Minimum-weight recovery was applied.
3. The residual operator was classified as:

```text
IL
XL
YL
ZL
```

This produces an exact ideal logical Pauli channel.

Representative logical error probabilities:

| Physical `p` | Logical `pL` |
|---:|---:|
| 0.001 | 0.00000998 |
| 0.002 | 0.00003982 |
| 0.005 | 0.00024723 |
| 0.010 | 0.00097796 |
| 0.020 | 0.00382505 |
| 0.050 | 0.02233185 |

The low-noise behavior is approximately second order because all single-qubit Pauli errors are correctable.

---

# Ideal [[5,1,3]] CIFAR-10 Results

The ideal five-qubit logical channel provides strong protection.

| `p` | Unprotected | Ideal `[[5,1,3]]` |
|---:|---:|---:|
| 0.000 | 91.43% | 91.43% |
| 0.001 | 91.33% | 91.43% |
| 0.002 | 91.17% | 91.42% |
| 0.005 | 89.74% | 91.41% |
| 0.010 | 74.74% | 91.30% |
| 0.020 | 34.43% | 90.64% |
| 0.050 | 13.19% | 29.19% |

At:

```text
p = 0.02
```

ideal QEC preserves:

```text
90.64%
```

accuracy while the unprotected classifier falls to:

```text
34.43%
```

This demonstrates the potential of QEC when correction operations themselves are assumed ideal.

---

# Stage 9: Noisy Five-Qubit Stabilizer Extraction

A circuit-level noisy syndrome model was introduced with representative parameters:

```text
two-qubit error     p2Q     = 0.005
single-qubit error  p1Q     = 0.001
ancilla preparation pprep   = 0.001
measurement error   pmeas   = 0.010
recovery error      prec    = 0.005
```

A naive single-ancilla stabilizer extraction circuit creates a major problem:

```text
ancilla fault
    ↓
propagation through later CNOTs
    ↓
correlated data error
    ↓
possible logical failure
```

This is a classic hook-error mechanism.

Even at:

```text
external data error p = 0
```

the naive circuit produced:

```text
logical error pL ≈ 0.039094
```

and classifier accuracy of only approximately:

```text
15.10%
```

The correction circuit itself had become the dominant error source.

---

# Stage 10: Flagged Fault-Tolerant Error Correction

A flag ancilla was introduced to detect dangerous propagated faults during stabilizer extraction.

The flag circuit allows the decoder to distinguish correlated errors that may have originated from a single physical fault.

The implementation automatically enumerates single two-qubit Pauli faults in the flagged circuit and constructs flag-conditioned correction tables.

For the stabilizer:

```text
X Z Z X I
```

the detected single-fault classes include:

```text
IIIII
IIZXI
IXZXI
IYZXI
IZZXI
IIIXI
IIXXI
IIYXI
```

The flagged procedure substantially reduces the logical error floor.

At:

```text
p = 0
```

the logical error decreases from:

```text
Naive QEC:
pL = 0.039094
```

to:

```text
Flagged FTEC:
pL = 0.013274
```

which corresponds to an approximately:

```text
66% relative reduction
```

in QEC-induced logical error.

---

# Flagged FTEC CIFAR-10 Results

| `p` | Unprotected | Ideal5 | Naive noisy QEC | Flagged FTEC |
|---:|---:|---:|---:|---:|
| 0.000 | 91.43% | 91.43% | 15.10% | **59.21%** |
| 0.001 | 91.33% | 91.43% | 14.84% | **54.16%** |
| 0.002 | 91.17% | 91.42% | 15.11% | **54.95%** |
| 0.005 | 89.74% | 91.41% | 14.73% | **45.53%** |
| 0.010 | 74.74% | 91.30% | 14.19% | **35.20%** |
| 0.020 | 34.43% | 90.64% | 13.78% | **22.33%** |
| 0.050 | 13.19% | 29.19% | 12.04% | 12.51% |

The result shows that fault-tolerant circuit design can recover a large amount of performance lost to naive syndrome extraction.

However, the remaining logical-error floor is still significant.

---

# Adaptive Flagged-FTEC Branching

The flagged decoder performs additional extraction only when needed.

Representative branch probabilities:

| `p` | Flag trigger | Syndrome trigger | No trigger |
|---:|---:|---:|---:|
| 0.000 | 7.88% | 9.96% | 82.16% |
| 0.001 | 7.84% | 10.41% | 81.74% |
| 0.002 | 7.83% | 10.83% | 81.35% |
| 0.005 | 7.77% | 12.19% | 80.04% |
| 0.010 | 7.64% | 14.16% | 78.20% |
| 0.020 | 7.51% | 18.29% | 74.20% |
| 0.050 | 6.96% | 29.28% | 63.76% |

At low physical noise, most correction cycles therefore avoid unnecessary follow-up circuitry.

---

# Stage 11: Syndrome-History Decoding

Repeated syndrome extraction was then studied using temporal information.

Three syndrome measurements were recorded:

```text
s(1)
s(2)
s(3)
```

Detection events were defined as:

```text
d(1) = s(1)
d(2) = s(2) XOR s(1)
d(3) = s(3) XOR s(2)
```

The decoder input contains:

```text
12 syndrome-history bits
+
4 flag-summary bits
=
16-bit history key
```

The final physical Pauli state belongs to one of:

```text
16 syndrome classes × 4 logical classes = 64 cosets
```

A noise-model-specific maximum-likelihood lookup decoder was calibrated using synthetic QEC circuit data.

Importantly:

```text
CIFAR-10 labels were NOT used to train the decoder.
```

---

# Syndrome-History Result

Mean logical errors across the tested noise range were:

```text
Adaptive flagged FTEC : 0.023167
History decoder       : 0.062830
Raw majority decoder  : 0.083861
```

Thus:

```text
History decoder > raw majority voting
```

in terms of logical performance, but:

```text
Adaptive flagged FTEC > history decoder
```

because three complete noisy syndrome rounds introduce substantially more circuit-level faults.

This demonstrates an important FTQC tradeoff:

```text
More syndrome information
        versus
More fault opportunities
```

---

# Stage 12: Gate-Aware Logical Error Model

The classifier contains many operations, and logical operations do not all have identical fault-tolerant implementation cost.

A gate-aware model was therefore introduced.

The final configuration uses:

```text
Rotation gadget cycles = 2
CNOT gadget cycles     = 2
```

This is a **logical-gadget overhead proxy**.

It is not an exact physical implementation of arbitrary logical `RY`, `RZ`, or a complete pieceable logical CNOT.

The uniform model has:

```text
160 logical fault locations
```

while the gate-aware model corresponds to:

```text
320 EC-equivalent protected channel applications
```

per inference.

---

# Final Decoder Selection

One decoder is selected globally for the full noise sweep instead of choosing a different decoder at every value of `p`.

Mean logical error:

| Decoder | Mean logical error |
|---|---:|
| Adaptive flagged FTEC | **0.023167** |
| Syndrome history | 0.062830 |
| 3-round majority | 0.083861 |

Therefore the final model uses:

```text
Adaptive flagged FTEC
```

as the fixed decoder.

---

# Final Logical-Gadget Error Model

Using the selected decoder:

| Physical `p` | Base protected `pL` | 2-cycle gadget `pL` | Expected logical fault events |
|---:|---:|---:|---:|
| 0.000 | 0.013274 | 0.026313 | 4.210 |
| 0.001 | 0.014274 | 0.028276 | 4.524 |
| 0.002 | 0.014294 | 0.028316 | 4.530 |
| 0.005 | 0.016382 | 0.032406 | 5.185 |
| 0.010 | 0.019848 | 0.039171 | 6.267 |
| 0.020 | 0.027324 | 0.053653 | 8.584 |
| 0.050 | 0.056772 | 0.109247 | 17.479 |

A key observation is that even at:

```text
p = 0
```

the noisy QEC circuit itself creates a nonzero logical-error floor.

The full classifier therefore experiences approximately:

```text
4.21 expected non-identity logical events
```

per inference in the final gate-aware model.

---

# Final End-to-End Results

The final experiment compares:

```text
Unprotected
UniformProtected
FinalFT
```

where:

- `Unprotected` applies physical depolarizing noise directly.
- `UniformProtected` applies the selected logical QEC channel once per modeled logical fault location.
- `FinalFT` additionally accounts for gate-aware logical-operation overhead.

Results:

| `p` | Unprotected | UniformProtected | FinalFT |
|---:|---:|---:|---:|
| 0.000 | `0.9143 ± 0.0000` | `0.5890 ± 0.0039` | `0.2352 ± 0.0017` |
| 0.001 | `0.9132 ± 0.0004` | `0.5434 ± 0.0019` | `0.2108 ± 0.0025` |
| 0.002 | `0.9115 ± 0.0007` | `0.5410 ± 0.0036` | `0.2087 ± 0.0027` |
| 0.005 | `0.8959 ± 0.0024` | `0.4535 ± 0.0036` | `0.1802 ± 0.0027` |
| 0.010 | `0.7483 ± 0.0019` | `0.3478 ± 0.0047` | `0.1506 ± 0.0031` |
| 0.020 | `0.3419 ± 0.0038` | `0.2190 ± 0.0035` | `0.1259 ± 0.0031` |
| 0.050 | `0.1299 ± 0.0017` | `0.1240 ± 0.0020` | `0.1181 ± 0.0032` |

---

# 77% Application-Level Target

An application-level target of:

```text
77% CIFAR-10 accuracy
```

was also evaluated.

The unprotected classifier remains above the target through the tested point:

```text
p = 0.005
```

where it achieves approximately:

```text
89.59%
```

At:

```text
p = 0.010
```

the unprotected model drops to:

```text
74.83%
```

and no longer meets the target.

The final gate-aware FT model does not reach the `77%` target at any tested error probability.

No FinalFT/unprotected application-level break-even point was observed over:

```text
0 ≤ p ≤ 0.05
```

---

# Main Finding

The experiments progressively change the conclusion as increasingly realistic fault-tolerance costs are introduced.

```text
Ideal QEC
    ↓
Strong error suppression

Naive noisy QEC
    ↓
Large QEC-induced logical error floor

Flagged FTEC
    ↓
Major reduction in correlated fault propagation

Repeated syndrome history
    ↓
Better decoding information but excessive extraction overhead

Gate-aware FT model
    ↓
Logical-operation overhead dominates the deep classifier
```

The main result is therefore:

> **Strong block-level error correction is not sufficient for application-level fault tolerance. The logical error per protected operation must be sufficiently small relative to the total number of protected operations in the algorithm.**

A useful approximate design intuition is:

```text
Nlogical × pL << 1
```

where:

- `Nlogical` is the number of relevant protected logical fault opportunities, and
- `pL` is the effective logical error probability per protected operation.

For circuits containing hundreds of protected logical locations, a logical error probability near `10^-2` can still be far too large.

---

# Conceptual Resource Estimate

The final model uses:

```text
8 logical qubits
```

protected by the `[[5,1,3]]` code:

```text
8 × 5 = 40 physical data qubits
```

If one syndrome ancilla and one flag ancilla are assigned to each block simultaneously:

```text
8 × 2 = 16 ancillas
```

giving a conceptual estimate of:

```text
40 + 16 = 56 active qubits
```

This is **not an exact compiled hardware requirement**.

Actual hardware requirements depend on:

- ancilla reuse,
- device connectivity,
- routing,
- logical-gate construction,
- intermediate error correction,
- physical native gate set,
- compiler decisions.

---

# Repository Structure

A typical repository layout is:

```text
fault-tolerant-quantum/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── basic_qec/
│   └── ...
│
├── fault_tolerance_test/
│   ├── figures/
│   ├── results/
│   └── ...
│
├── figures/
│   └── generated experimental figures
│
├── results/
│   ├── CSV result files
│   └── text summaries
│
├── cifar10_pca8_baseline.py
├── cifar10_quantum_head.py
├── cifar10_quantum_head_batched.py
├── cifar10_circuit_level_qec.py
├── cifar10_flagged_5qubit_ftec.py
├── cifar10_5qubit_syndrome_history.py
├── cifar10_ft_logical_gate_model.py
├── cifar10_final_end_to_end_ft.py
└── ...
```

Earlier exploratory scripts are retained because they document the progression from basic QEC to the final FT model.

---

# Important Scripts

## `cifar10_pca8_baseline.py`

Initial low-dimensional CIFAR-10 baseline using PCA-reduced features.

This experiment produced approximately `30%` test accuracy and motivated the switch to stronger learned image representations.

---

## `cifar10_quantum_head.py`

Initial hybrid quantum-classical classifier.

---

## `cifar10_quantum_head_batched.py`

Batched/GPU implementation of the trained 8-qubit quantum classifier.

The resulting trained checkpoint reaches approximately:

```text
91.43% CIFAR-10 test accuracy
```

under ideal simulation.

---

## `cifar10_circuit_level_qec.py`

Evaluates repetition-code QEC under explicit circuit-level errors such as noisy syndrome CNOTs, measurements, and recovery.

---

## `cifar10_flagged_5qubit_ftec.py`

Implements the flagged `[[5,1,3]]` syndrome-extraction experiment.

Main functions include:

- noisy stabilizer extraction,
- flag ancilla simulation,
- single-fault enumeration,
- flag-conditioned correction,
- Monte Carlo logical-channel estimation,
- CIFAR-10 evaluation.

Important outputs include:

```text
results/cifar10_flagged_5qubit_ftec_detailed.csv
results/cifar10_flagged_5qubit_ftec_summary.csv
figures/cifar10_flagged_5qubit_ftec_accuracy.png
figures/five_qubit_flagged_ftec_logical_error.png
figures/cifar10_flagged_ftec_gain.png
figures/five_qubit_flagged_ftec_branch_rates.png
```

---

## `cifar10_5qubit_syndrome_history.py`

Evaluates repeated flagged syndrome extraction using temporal detection events.

The decoder uses:

```text
3 syndrome rounds
12 temporal syndrome bits
4 flag-summary bits
16-bit history key
64 possible final Pauli cosets
```

The decoder is calibrated from the circuit-level noise model rather than CIFAR-10 labels.

Important outputs include:

```text
results/five_qubit_syndrome_history_channels.csv
results/cifar10_syndrome_history_detailed.csv
results/cifar10_syndrome_history_summary.csv
figures/cifar10_syndrome_history_accuracy.png
figures/five_qubit_syndrome_history_logical_error.png
figures/cifar10_syndrome_history_gain.png
figures/five_qubit_history_decoder_coset_accuracy.png
figures/five_qubit_syndrome_history_activity.png
```

---

## `cifar10_ft_logical_gate_model.py`

Introduces gate-specific logical overhead instead of assigning the same logical channel to every operation.

It evaluates:

```text
Unprotected
Uniform history-channel protection
Gate-aware FT logical model
```

and studies sensitivity to the assumed logical rotation overhead.

---

## `cifar10_final_end_to_end_ft.py`

Final simulator-stage experiment.

It:

1. Loads the decoder comparison results.
2. Selects one fixed decoder globally.
3. Uses the best decoder for the entire physical-noise sweep.
4. Applies the final logical Pauli channel.
5. Includes gate-aware FT overhead.
6. Evaluates CIFAR-10 accuracy and Macro-F1.
7. Computes per-class recall.
8. Generates confusion matrices.
9. Computes application-level target and break-even statistics.
10. Saves the final resource summary.

Primary outputs:

```text
results/cifar10_final_ft_detailed.csv
results/cifar10_final_ft_summary.csv
results/cifar10_final_ft_per_class.csv
results/cifar10_final_ft_gadget_rates.csv
results/cifar10_final_ft_resource_report.txt
```

Figures include:

```text
figures/cifar10_final_ft_accuracy.png
figures/cifar10_final_ft_gain_vs_unprotected.png
figures/cifar10_final_ft_overhead_penalty.png
figures/cifar10_final_ft_gadget_error_rates.png
figures/cifar10_final_ft_expected_logical_events.png
```

---

# Installation

The experiments were developed and tested using:

```text
Python 3.10
PyTorch 2.14.0+cu126
CUDA 12.6
NVIDIA GeForce RTX 4090
```

A CUDA-capable GPU is strongly recommended for the full CIFAR-10 Monte Carlo experiments.

Clone the repository:

```bash
git clone git@github.com:Ratun11/fault-tolerant-quantum.git
cd fault-tolerant-quantum
```

Create a virtual environment:

```bash
python3 -m venv braket-ftqc
source braket-ftqc/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` has not yet been prepared, the main packages used in the project include:

```text
torch
torchvision
numpy
scikit-learn
matplotlib
pennylane
```

Additional Amazon Braket dependencies will be required for the hardware-validation stage.

---

# Dataset

This project uses the standard **CIFAR-10** dataset.

The raw dataset is intentionally not included in the GitHub repository.

The local repository may contain:

```text
data/
```

but this directory should be excluded through `.gitignore`.

For reproducibility, CIFAR-10 can be downloaded through `torchvision` or from the official dataset source.

---

# Required Intermediate Files

Some later experiments depend on intermediate outputs from earlier stages.

For example, the trained quantum classifier uses files similar to:

```text
results/cifar10_resnet32_features.npz
results/cifar10_quantum_batched_best.pt
```

These files may be excluded from GitHub if they are large.

If they are not included in the repository, users must regenerate them by running the corresponding feature-extraction and training scripts before executing later FT experiments.

The final logical-gate simulations also depend on:

```text
results/five_qubit_syndrome_history_channels.csv
```

which is produced by:

```bash
python cifar10_5qubit_syndrome_history.py
```

---

# Typical Experiment Order

For reproducing the main FTQC pipeline, the recommended progression is approximately:

```bash
# Train / prepare the hybrid classifier
python cifar10_quantum_head_batched.py

# Earlier QEC experiments
python cifar10_circuit_level_qec.py

# Flagged FTEC
python cifar10_flagged_5qubit_ftec.py

# Syndrome-history decoding
python cifar10_5qubit_syndrome_history.py

# Logical-gate overhead study
python cifar10_ft_logical_gate_model.py

# Final end-to-end experiment
python cifar10_final_end_to_end_ft.py
```

The complete repository contains additional intermediate and diagnostic experiments that were used to build the final model.

---

# Monte Carlo Configuration

Several classifier-level experiments use:

```text
32 stochastic trajectories per sample
5 random seeds
10,000 CIFAR-10 test samples
```

Large encoded-block channel calculations may use hundreds of thousands of Monte Carlo samples.

For example, the syndrome-history decoder used approximately:

```text
100,000 calibration samples per physical noise value
300,000 independent evaluation samples per physical noise value
```

The exact configuration is defined inside each experiment script.

---

# Figures

The `figures/` directory contains plots produced during the study, including:

- quantum-noise sensitivity,
- logical-vs-physical error scaling,
- ideal QEC recovery,
- noisy syndrome comparisons,
- QEC break-even heatmaps,
- five-qubit logical error probabilities,
- logical Pauli distributions,
- flagged-FTEC performance,
- flag/syndrome branch statistics,
- syndrome-history decoder performance,
- logical-gate overhead,
- final end-to-end FT accuracy,
- expected logical fault events.

These figures correspond to the experimental progression described above.

---

# Results

The `results/` directory contains numerical outputs such as:

```text
CSV files
logical channel estimates
per-seed classifier results
summary statistics
resource estimates
decoder comparison results
```

The final results are stored primarily in:

```text
results/cifar10_final_ft_detailed.csv
results/cifar10_final_ft_summary.csv
results/cifar10_final_ft_per_class.csv
results/cifar10_final_ft_gadget_rates.csv
results/cifar10_final_ft_resource_report.txt
```

---

# Scientific Scope and Limitations

This repository intentionally distinguishes between several levels of modeling.

## 1. Logical statevector simulation

The final classifier simulates:

```text
8 logical qubits
```

rather than a full statevector of all encoded physical qubits.

The `[[5,1,3]]` physical blocks are represented using stabilizer/Pauli-frame calculations.

Therefore this repository does **not** simulate a complete 40- or 56-qubit encoded statevector.

## 2. Conceptual 56-qubit estimate

The approximate:

```text
56-qubit
```

resource count comes from:

```text
40 encoded data qubits
+
16 syndrome/flag ancillas
```

assuming blockwise parallel correction.

It is a conceptual resource estimate, not an exact compiled hardware circuit size.

## 3. Logical-gate model

The final arbitrary logical `RY` and `RZ` operations are represented through logical-gadget overhead cycles.

They are **not** explicitly compiled into a complete universal fault-tolerant gate construction such as a full Clifford+T / magic-state implementation.

Similarly, the logical CNOT model is an overhead proxy and does not explicitly simulate every correlated intermediate state of a complete pieceable FT entangling-gate construction.

## 4. Noise model

Most physical faults are represented through stochastic Pauli/depolarizing channels.

Real hardware may also contain:

```text
coherent errors
amplitude damping
dephasing bias
leakage
crosstalk
time-dependent drift
correlated noise
```

These effects are outside the current simulation model.

## 5. No quantum-advantage claim

The classifier uses a pretrained classical ResNet18 for feature extraction.

Therefore the reported CIFAR-10 accuracy should not be interpreted as evidence of quantum computational advantage or superior performance relative to classical image classifiers.

The classifier is used as a realistic downstream workload for studying accumulated quantum errors and fault-tolerance overhead.

---

# Interpretation of the Negative Final Result

The final FT model does not outperform the unprotected classifier in the tested noise range.

This is an important result rather than a simulation failure.

Earlier experiments show:

```text
ideal QEC → strong protection
```

while increasingly realistic models show:

```text
noisy extraction
+ fault propagation
+ repeated syndrome overhead
+ logical-gate overhead
→ loss of the ideal-QEC benefit
```

This identifies the actual bottleneck:

> The remaining logical error per protected operation is still too large relative to the number of protected operations required by the classifier.

The study therefore emphasizes the difference between:

```text
error correction
```

and:

```text
application-level fault tolerance
```

---

# Hardware Validation

The next stage of the project is hardware validation using **Amazon Braket** and a **Rigetti QPU**.

The intended hardware workflow is:

```text
Trained 8-qubit classifier
        ↓
Finite-shot local simulation
        ↓
Amazon Braket simulator validation
        ↓
Rigetti topology mapping
        ↓
Small CIFAR-10 test subset
        ↓
Hardware execution
```

The full conceptual 56-qubit FT classifier will not initially be presented as a direct hardware implementation.

Instead, hardware experiments will focus on controlled components such as:

```text
8-qubit unencoded classifier inference
flagged stabilizer extraction
selected QEC gadgets
finite-shot expectation estimation
```

This maintains a clear distinction between simulated FT architecture and physically executed circuits.

---

# Reproducibility

Random seeds are fixed in the main scripts where possible.

The final experiments use:

```text
SEED = 42
```

with deterministic seed offsets for repeated classifier trials.

Exact results can still vary slightly depending on:

```text
PyTorch version
CUDA version
GPU architecture
Monte Carlo sampling
library versions
```

---

# Development Environment

The project was developed primarily using:

```text
Visual Studio Code
Python
PyTorch
CUDA
PennyLane
NumPy
scikit-learn
Matplotlib
```

Primary local hardware:

```text
NVIDIA GeForce RTX 4090
```

---

# Repository Status

Current status:

```text
[✓] Basic QEC
[✓] Repetition-code QEC
[✓] Noisy syndrome extraction
[✓] Fault-propagation analysis
[✓] CIFAR-10 hybrid classifier
[✓] GPU quantum simulator
[✓] General Pauli noise
[✓] [[5,1,3]] ideal QEC
[✓] Circuit-level five-qubit QEC
[✓] Flagged FTEC
[✓] Syndrome-history decoder
[✓] Gate-aware logical model
[✓] Final end-to-end FT simulation
[ ] Amazon Braket validation
[ ] Rigetti QPU experiments
```


---

# License

A license has not yet been specified.

Before public release, consider adding an open-source license such as the MIT License or Apache License 2.0, depending on how you want the code to be reused.

---

# Contact

For questions about the code or experiments, please open a GitHub issue in this repository.

Repository:

```text
https://github.com/Ratun11/fault-tolerant-quantum
```

---

## Summary

This repository demonstrates a complete progression from elementary QEC concepts to an application-level fault-tolerant quantum machine-learning simulation.

The most important observation is that:

> **An error-correcting code may strongly suppress isolated physical errors while still failing to provide an application-level advantage when noisy correction and logical-gate overhead are repeated throughout a deep quantum circuit.**

The results therefore motivate future work on lower-overhead QEC, adaptive correction schedules, Pauli-frame recovery, hardware-aware logical gates, and FT-aware quantum machine-learning circuit design.
