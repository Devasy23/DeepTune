import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.applications import resnet50 as resnet

class TripletDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, image_size, num_classes):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        self.image_data_generator = ImageDataGenerator(preprocessing_function=resnet.preprocess_input)
        self.on_epoch_end()
        print(f"Initialized TripletDataGenerator with {len(self.image_paths)} images")

    def __len__(self):
        return max(1, len(self.image_paths) // self.batch_size)  # Ensure at least one batch

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.encoded_labels[index * self.batch_size:(index + 1) * self.batch_size]
        return self._generate_triplet_batch(batch_image_paths, batch_labels)

    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        combined = list(zip(self.image_paths, self.encoded_labels))
        np.random.shuffle(combined)
        self.image_paths[:], self.encoded_labels[:] = zip(*combined)

    def _generate_triplet_batch(self, batch_image_paths, batch_labels):
        anchor_images = []
        positive_images = []
        negative_images = []

        for i in range(len(batch_image_paths)):
            anchor_path = batch_image_paths[i]
            anchor_label = batch_labels[i]

            positive_path = np.random.choice(
                [p for p, l in zip(self.image_paths, self.encoded_labels) if l == anchor_label]
            )
            negative_path = np.random.choice(
                [p for p, l in zip(self.image_paths, self.encoded_labels) if l != anchor_label]
            )

            anchor_image = load_img(anchor_path, target_size=self.image_size)
            positive_image = load_img(positive_path, target_size=self.image_size)
            negative_image = load_img(negative_path, target_size=self.image_size)

            anchor_images.append(img_to_array(anchor_image))
            positive_images.append(img_to_array(positive_image))
            negative_images.append(img_to_array(negative_image))

        # Convert lists to numpy arrays
        anchor_array = np.array(anchor_images)
        positive_array = np.array(positive_images)
        negative_array = np.array(negative_images)

        # Return inputs and dummy targets (zeros) since the loss is computed in the model
        return (
            {
                "anchor": anchor_array,
                "positive": positive_array,
                "negative": negative_array
            },
            np.zeros((len(batch_image_paths),))  # Dummy target values
        )