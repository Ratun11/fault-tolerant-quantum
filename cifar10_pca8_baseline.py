import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "results/cifar10_pca8.npz"

FIGURE_DIR = "figures"
RESULT_DIR = "results"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD PROCESSED CIFAR-10
# ============================================================

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X_train = data["X_train"]
y_train = data["y_train"]

X_test = data["X_test"]
y_test = data["y_test"]

class_names = data["class_names"]


print("\n======================================")
print("DATASET")
print("======================================")

print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)

print("Classes:")
print(class_names)


# ============================================================
# 2. CLASSICAL MULTICLASS BASELINE
#
# This uses exactly the same 8 features that
# will later enter the quantum circuit.
# ============================================================

print("\n======================================")
print("TRAINING CLASSICAL BASELINE")
print("======================================")

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 3. PREDICTIONS
# ============================================================

train_predictions = model.predict(
    X_train
)

test_predictions = model.predict(
    X_test
)


train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print("\n======================================")
print("RESULTS")
print("======================================")

print(
    f"Training accuracy = "
    f"{train_accuracy:.4f}"
)

print(
    f"Test accuracy     = "
    f"{test_accuracy:.4f}"
)


# ============================================================
# 4. PER-CLASS RESULTS
# ============================================================

report = classification_report(
    y_test,
    test_predictions,
    target_names=class_names,
    output_dict=True
)


print("\n======================================")
print("PER-CLASS ACCURACY")
print("======================================")

per_class_recall = []

for class_name in class_names:

    recall = report[str(class_name)]["recall"]

    per_class_recall.append(
        recall
    )

    print(
        f"{class_name:12s}: "
        f"{recall:.4f}"
    )


# ============================================================
# 5. SAVE SUMMARY CSV
# ============================================================

csv_path = os.path.join(
    RESULT_DIR,
    "cifar10_pca8_classical_baseline.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "metric",
        "value"
    ])

    writer.writerow([
        "train_accuracy",
        train_accuracy
    ])

    writer.writerow([
        "test_accuracy",
        test_accuracy
    ])

    for class_name, recall in zip(
        class_names,
        per_class_recall
    ):

        writer.writerow([
            f"{class_name}_recall",
            recall
        ])


print(
    "\nResults saved to:",
    csv_path
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)


plt.figure(
    figsize=(9, 8)
)

plt.imshow(cm)

plt.title(
    "CIFAR-10 PCA-8 Classical Baseline"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.xticks(
    range(10),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(10),
    class_names
)

plt.colorbar()


# Add values inside cells
for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=7
        )


plt.tight_layout()


cm_path = os.path.join(
    FIGURE_DIR,
    "cifar10_pca8_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Confusion matrix saved to:",
    cm_path
)


# ============================================================
# 7. PER-CLASS PERFORMANCE FIGURE
# ============================================================

plt.figure(
    figsize=(10, 5)
)

bars = plt.bar(
    class_names,
    np.array(per_class_recall) * 100
)


for bar, value in zip(
    bars,
    per_class_recall
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{value * 100:.1f}%",

        ha="center",
        va="bottom",
        fontsize=8
    )


plt.ylabel(
    "Recall (%)"
)

plt.xlabel(
    "CIFAR-10 Class"
)

plt.title(
    "PCA-8 Baseline Performance by CIFAR-10 Class"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()


class_path = os.path.join(
    FIGURE_DIR,
    "cifar10_pca8_per_class.png"
)

plt.savefig(
    class_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Per-class figure saved to:",
    class_path
)