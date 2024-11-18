import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

class TripletGenerator(Sequence):
    """Data generator for triplet-based training."""
    
    def __init__(self, image_paths, labels, batch_size, image_size, preprocessing_fn=None):
        """Initialize the generator.
        
        Args:
            image_paths: List of image paths
            labels: List of corresponding labels
            batch_size: Batch size
            image_size: Tuple of (height, width)
            preprocessing_fn: Optional preprocessing function for the images
        """
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        self.on_epoch_end()
        
    def __len__(self):
        return max(1, len(self.image_paths) // self.batch_size)
    
    def __getitem__(self, index):
        batch_indices = range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.image_paths)))
        return self._generate_triplet_batch(batch_indices)
    
    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.encoded_labels = self.encoded_labels[indices]
    
    def _load_image(self, image_path):
        """Load and preprocess an image."""
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img)
        
        if self.preprocessing_fn:
            img_array = self.preprocessing_fn(img_array)
            
        return img_array
    
    def _generate_triplet_batch(self, batch_indices):
        """Generate a batch of triplets (anchor, positive, negative)."""
        anchors = []
        positives = []
        negatives = []
        
        for idx in batch_indices:
            anchor_label = self.encoded_labels[idx]
            anchor_path = self.image_paths[idx]
            
            # Find positive (same label) and negative (different label) samples
            positive_paths = [p for i, p in enumerate(self.image_paths)
                            if self.encoded_labels[i] == anchor_label and p != anchor_path]
            negative_paths = [p for i, p in enumerate(self.image_paths)
                            if self.encoded_labels[i] != anchor_label]
            
            if not positive_paths or not negative_paths:
                continue
                
            positive_path = np.random.choice(positive_paths)
            negative_path = np.random.choice(negative_paths)
            
            # Load images
            anchors.append(self._load_image(anchor_path))
            positives.append(self._load_image(positive_path))
            negatives.append(self._load_image(negative_path))
        
        # Convert lists to numpy arrays
        anchors = np.array(anchors)
        positives = np.array(positives)
        negatives = np.array(negatives)
        
        # Create dummy target (not used in training)
        dummy_target = np.zeros((len(anchors),))
        
        return {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives
        }, dummy_target

def load_image_dataset(data_dir, image_size, test_split=0.2, seed=None):
    """Load image dataset from directory.
    
    Args:
        data_dir: Directory containing class subdirectories
        image_size: Tuple of (height, width)
        test_split: Fraction of data to use for testing
        seed: Random seed
    
    Returns:
        Tuple of (train_paths, train_labels, test_paths, test_labels)
    """
    image_paths = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_paths.append(os.path.join(label_dir, image_name))
                labels.append(label)
    
    # Split data
    if seed is not None:
        np.random.seed(seed)
        
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_split))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_paths, train_labels, test_paths, test_labels
