from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def show_samples(dataset_path: Path, classes: list[str], num_samples=2):
    """
    Display a specific number of images from each class using matplotlib.
    Note:
        Each class must have its own subdirectory under `dataset_path`.
        Ignores files containing 'mask' in their name.
    """
    plt.figure(figsize=(len(classes) * 3, num_samples * 3))
    for i, cls_name in enumerate(classes):
        class_dir = dataset_path / Path(cls_name)
        displayed = 0

        for fname in class_dir.iterdir():
            if fname.name.endswith("mask.png"):
                continue

            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                plt.subplot(num_samples, len(classes),
                            displayed * len(classes) + i + 1)
                plt.imshow(img, cmap='gray')
                plt.title(cls_name.capitalize())
                plt.axis('off')
                displayed += 1

            if displayed >= num_samples:
                break

    plt.tight_layout()
    plt.show()


def plot_class_distribution(dataset_path: Path, classes: list[str]):
    """
    Plot the number of images per class (excluding mask files).
    Note:
        Assumes each class has its own subdirectory under `dataset_path`.
        Files containing 'mask' in the name are excluded.
    """
    counts = []
    for cls in classes:
        class_dir = dataset_path / Path(cls)
        images = [img for img in class_dir.iterdir() if 'mask' not in img.name]
        counts.append(len(images))

    plt.figure(figsize=(6, 4))
    bars = plt.bar(classes, counts, color='mediumpurple')

    for bar_chart in bars:
        height = bar_chart.get_height()
        plt.text(bar_chart.get_x() + bar_chart.get_width() / 2, height,
                 str(height), ha='center', va='bottom', fontsize=10)

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()
