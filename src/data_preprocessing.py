import cv2
import numpy as np

def load_ultrasound_images(dataset_path, classes, image_size=224):
    X, y = [], []
    for i, cls_name in enumerate(classes):
        class_dir = dataset_path / cls_name
        for image_file in class_dir.iterdir():
            if ('mask' in image_file.name or
                not image_file.suffix.lower() in ['.png', '.jpg', '.jpeg']):
                continue
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (image_size, image_size))
            image = np.stack([image] * 3, axis=-1)
            X.append(image)
            y.append(i)
    return np.array(X), np.array(y)
