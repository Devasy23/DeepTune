import tensorflow as tf
import numpy as np
from pathlib import Path
import random


class ArcFaceDataGenerator:
    def __init__(self, data_dir, batch_size=32, image_size=(112, 112), 
                 augment=True, cache=True):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.cache = cache
        
        # Create class mapping
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.class_names)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare image paths and labels."""
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)
    
    def _parse_image(self, image_path):
        """Load and preprocess image."""
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Resize
        img = tf.image.resize(img, self.image_size)
        
        # Normalize to [-1, 1]
        img = (tf.cast(img, tf.float32) - 127.5) / 128.0
        
        return img
    
    def _augment_image(self, image):
        """Apply data augmentation."""
        if random.random() > 0.5:
            image = tf.image.random_flip_left_right(image)
        
        # Add more augmentations as needed
        # image = tf.image.random_brightness(image, 0.2)
        # image = tf.image.random_contrast(image, 0.8, 1.2)
        
        return image
    
    def create_dataset(self, is_training=True):
        """Create tf.data.Dataset for training or validation."""
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))
        
        if is_training:
            # Shuffle if training
            dataset = dataset.shuffle(buffer_size=len(self.image_paths))
        
        # Map functions
        dataset = dataset.map(
            lambda x, y: (self._parse_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training and self.augment:
            dataset = dataset.map(
                lambda x, y: (self._augment_image(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        if self.cache:
            dataset = dataset.cache()
            
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @property
    def steps_per_epoch(self):
        """Calculate steps per epoch based on dataset size and batch size."""
        return len(self.image_paths) // self.batch_size
