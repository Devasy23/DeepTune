import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(224, 224)):
    """
    Load and preprocess data from the given directory.

    Args:
        data_dir: The directory containing the dataset.
        img_size: The target size of the images.

    Returns:
        A tuple of (images, labels).
    """
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0  # Normalize the image
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def create_triplets(images, labels):
    """
    Create triplets of images (anchor, positive, negative) for training.

    Args:
        images: The array of images.
        labels: The array of labels.

    Returns:
        A tuple of (anchor_images, positive_images, negative_images).
    """
    anchor_images = []
    positive_images = []
    negative_images = []
    num_classes = len(np.unique(labels))
    for i in range(len(images)):
        anchor = images[i]
        anchor_label = labels[i]
        positive_indices = np.where(labels == anchor_label)[0]
        negative_indices = np.where(labels != anchor_label)[0]
        positive = images[np.random.choice(positive_indices)]
        negative = images[np.random.choice(negative_indices)]
        anchor_images.append(anchor)
        positive_images.append(positive)
        negative_images.append(negative)
    return np.array(anchor_images), np.array(positive_images), np.array(negative_images)

def preprocess_data(data_dir, img_size=(224, 224), test_size=0.2):
    """
    Preprocess data for training and testing.

    Args:
        data_dir: The directory containing the dataset.
        img_size: The target size of the images.
        test_size: The proportion of the dataset to include in the test split.

    Returns:
        A tuple of (train_triplets, test_triplets).
    """
    images, labels = load_data(data_dir, img_size)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, stratify=labels)
    train_triplets = create_triplets(train_images, train_labels)
    test_triplets = create_triplets(test_images, test_labels)
    return train_triplets, test_triplets

def augment_data(images, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True):
    """
    Augment the images to increase the diversity of the training data.

    Args:
        images: The array of images to augment.
        rotation_range: The range of rotation for augmentation.
        width_shift_range: The range of width shift for augmentation.
        height_shift_range: The range of height shift for augmentation.
        horizontal_flip: Whether to randomly flip images horizontally.

    Returns:
        The augmented images.
    """
    datagen = ImageDataGenerator(rotation_range=rotation_range,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 horizontal_flip=horizontal_flip)
    augmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)
        aug_iter = datagen.flow(img)
        aug_img = next(aug_iter)[0]
        augmented_images.append(aug_img)
    return np.array(augmented_images)
