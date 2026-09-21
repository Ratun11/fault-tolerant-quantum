import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

N_FEATURES = 8

DATA_DIR = "data"
RESULT_DIR = "results"
FIGURE_DIR = "figures"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# CIFAR-10 CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ============================================================
# 1. DOWNLOAD CIFAR-10
# ============================================================

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


print("\n======================================")
print("CIFAR-10")
print("======================================")

print("Training samples:", len(train_dataset))
print("Test samples:    ", len(test_dataset))
print("Number of classes:", len(CLASS_NAMES))


# ============================================================
# 2. CONVERT DATASET TO NUMPY
# ============================================================

def dataset_to_numpy(dataset):

    images = []
    labels = []

    for image, label in dataset:

        images.append(
            image.numpy()
        )

        labels.append(
            label
        )

    return (
        np.array(images),
        np.array(labels)
    )


X_train, y_train = dataset_to_numpy(
    train_dataset
)

X_test, y_test = dataset_to_numpy(
    test_dataset
)


print("\nOriginal training shape:")
print(X_train.shape)

print("\nOriginal test shape:")
print(X_test.shape)


# ============================================================
# 3. SHOW ONE IMAGE FROM EACH CLASS
# ============================================================

plt.figure(
    figsize=(12, 5)
)

for class_id in range(10):

    index = np.where(
        y_train == class_id
    )[0][0]

    image = X_train[index]

    # PyTorch:
    # C x H x W
    #
    # Matplotlib:
    # H x W x C

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    plt.subplot(
        2,
        5,
        class_id + 1
    )

    plt.imshow(image)

    plt.title(
        CLASS_NAMES[class_id]
    )

    plt.axis("off")


plt.tight_layout()

class_figure_path = os.path.join(
    FIGURE_DIR,
    "cifar10_all_classes.png"
)

plt.savefig(
    class_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "\nClass figure saved to:",
    class_figure_path
)


# ============================================================
# 4. FLATTEN IMAGES
#
# 32 x 32 x 3 = 3072
# ============================================================

X_train_flat = X_train.reshape(
    len(X_train),
    -1
)

X_test_flat = X_test.reshape(
    len(X_test),
    -1
)


print("\nFlattened training shape:")
print(X_train_flat.shape)


# ============================================================
# 5. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_flat
)

X_test_scaled = scaler.transform(
    X_test_flat
)


# ============================================================
# 6. PCA
#
# 3072 dimensions -> 8 dimensions
# ============================================================

pca = PCA(
    n_components=N_FEATURES,
    random_state=42
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)


print("\n======================================")
print("PCA")
print("======================================")

print(
    "Reduced training shape:",
    X_train_pca.shape
)

print(
    "Reduced test shape:",
    X_test_pca.shape
)

print(
    "Explained variance ratio:"
)

print(
    pca.explained_variance_ratio_
)

print(
    "\nTotal explained variance:"
)

print(
    pca.explained_variance_ratio_.sum()
)


# ============================================================
# 7. SCALE PCA FEATURES FOR QUANTUM ROTATIONS
#
# We want approximately:
#
# -pi <= x <= pi
#
# for rotation gates later.
# ============================================================

max_abs = np.max(
    np.abs(X_train_pca),
    axis=0
)

# Avoid division by zero
max_abs[
    max_abs == 0
] = 1.0


X_train_quantum = (
    np.pi
    * X_train_pca
    / max_abs
)

X_test_quantum = (
    np.pi
    * X_test_pca
    / max_abs
)


print("\nQuantum feature range:")

print(
    X_train_quantum.min(),
    X_train_quantum.max()
)


# ============================================================
# 8. CLASS DISTRIBUTION
# ============================================================

class_counts = []

for class_id in range(10):

    count = np.sum(
        y_train == class_id
    )

    class_counts.append(
        count
    )


plt.figure(
    figsize=(9, 5)
)

bars = plt.bar(
    CLASS_NAMES,
    class_counts
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.ylabel(
    "Training Samples"
)

plt.xlabel(
    "CIFAR-10 Class"
)

plt.title(
    "CIFAR-10 Training Class Distribution"
)

plt.tight_layout()


distribution_path = os.path.join(
    FIGURE_DIR,
    "cifar10_class_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Distribution figure saved to:",
    distribution_path
)


# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

output_path = os.path.join(
    RESULT_DIR,
    "cifar10_pca8.npz"
)

np.savez_compressed(

    output_path,

    X_train=X_train_quantum,
    y_train=y_train,

    X_test=X_test_quantum,
    y_test=y_test,

    explained_variance=(
        pca.explained_variance_ratio_
    ),

    pca_max_abs=max_abs,

    class_names=np.array(
        CLASS_NAMES
    )
)


print("\n======================================")
print("DONE")
print("======================================")

print(
    "Processed dataset saved to:",
    output_path
)

print(
    "Training shape:",
    X_train_quantum.shape
)

print(
    "Test shape:",
    X_test_quantum.shape
)