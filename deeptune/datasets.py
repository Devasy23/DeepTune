import tensorflow_datasets as tfds

def load_dataset(dataset_name, split='train'):
    """
    Load a dataset using TensorFlow Datasets (TFDS).

    Args:
        dataset_name: The name of the dataset to load.
        split: The split of the dataset to load (e.g., 'train', 'test').

    Returns:
        A tf.data.Dataset object.
    """
    dataset, info = tfds.load(dataset_name, split=split, with_info=True)
    return dataset, info

def preprocess_dataset(dataset, img_size=(224, 224)):
    """
    Preprocess the dataset by resizing and normalizing the images.

    Args:
        dataset: The tf.data.Dataset object to preprocess.
        img_size: The target size of the images.

    Returns:
        A preprocessed tf.data.Dataset object.
    """
    def _preprocess(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_lfw_dataset(split='train'):
    """
    Get the LFW dataset.

    Args:
        split: The split of the dataset to load (e.g., 'train', 'test').

    Returns:
        A preprocessed tf.data.Dataset object.
    """
    dataset, info = load_dataset('lfw', split=split)
    dataset = preprocess_dataset(dataset)
    return dataset
