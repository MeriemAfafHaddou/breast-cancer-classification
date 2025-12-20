"""
Utilities for loading ultrasound images, balancing classes with data augmentation,
and generating stratified k-fold splits for training and validation.
"""

# pylint: disable=no-member
import cv2
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold


def load_ultrasound_images(dataset_path, classes, image_size=224):
    """
    Loads grayscale ultrasound images from subdirectories, resizes them, and
    stacks them into 3 channels for model compatibility.

    Each class must be in its own subdirectory under `dataset_path`.
    """
    x, y = [], []
    for i, cls_name in enumerate(classes):
        class_dir = dataset_path / cls_name
        for image_file in class_dir.iterdir():
            if "mask" in image_file.name or not image_file.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
            ]:
                continue
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (image_size, image_size))
            image = np.stack([image] * 3, axis=-1)
            x.append(image)
            y.append(i)
    return np.array(x), np.array(y)


def augment_class(
    x: np.ndarray,
    y: np.ndarray,
    class_idx: int,
    target_count: int,
    augmenter: ImageDataGenerator,
):
    """
    Augments samples of a class using the given ImageDataGenerator
    until it reaches the specified target_count.
    """
    x_class = x[y == class_idx]
    y_class = y[y == class_idx]

    # How many more samples we need
    n_to_generate = target_count - len(x_class)

    if n_to_generate <= 0:
        return x_class, y_class  # already balanced

    augmented = []
    for _ in range(n_to_generate):
        idx = np.random.randint(0, len(x_class))
        image = x_class[idx].reshape((1, *x_class[idx].shape))
        aug_img = next(augmenter.flow(image, batch_size=1))[0]
        augmented.append(aug_img.astype(np.uint8))

    x_aug = np.array(augmented)
    y_aug = np.full(n_to_generate, class_idx)

    return np.concatenate([x_class, x_aug]), np.concatenate([y_class, y_aug])


# pylint: disable=too-many-locals
def k_fold_augment(x, y, n_splits=10, augmenter=None):
    """
    Performs 10-fold stratified cross-validation and augments the training set
    in each fold to balance classes.

    Returns a list of tuples:
    [(x_train_fold, y_train_fold, x_val_fold, y_val_fold), ...]
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        print(f"\n🔁 Fold {fold_idx + 1}/{n_splits}")

        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        # Get target count (max class count for balancing)
        classes, counts = np.unique(y_train, return_counts=True)
        target_count = max(counts)

        # Augment each class in the training fold
        x_balanced, y_balanced = [], []
        print(classes)
        for class_idx in classes:
            x_aug_class, y_aug_class = augment_class(
                x_train, y_train, class_idx, target_count, augmenter
            )
            x_balanced.append(x_aug_class)
            y_balanced.append(y_aug_class)
            print(f"Class {class_idx} original shape: {x_train.shape}")
            print(f"Class {class_idx} augmented shape: {x_aug_class.shape}")

        x_train_balanced = np.concatenate(x_balanced)
        y_train_balanced = np.concatenate(y_balanced)

        # Shuffle after augmentation
        indices = np.arange(len(x_train_balanced))
        np.random.shuffle(indices)
        x_train_balanced = x_train_balanced[indices]
        y_train_balanced = y_train_balanced[indices]

        folds.append((x_train_balanced, y_train_balanced, x_val, y_val))

    return folds
