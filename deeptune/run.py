from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb
import os
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.metrics import Mean
from keras.layers import Input
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from deeptune.utils import get_embedding_module, get_siamese_network
from deeptune.siamese_network import SiameseModel
from deeptune.tripletdatagenerator import TripletDataGenerator

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

data_dir = 'datasets/lfw_processed'
image_size = (224, 224)
batch_size = 2  # Adjust the batch size for the small dataset
margin = 10.0

# Initialize W&B
wandb.init(project="FaceRec", config={
    "learning_rate": 0.001,
    "epochs": 20,
    "batch_size": 2,
    "optimizer": "Adam",
    "architecture": "ResNet50",
    "dataset": "lfw",
    "loss": "TripletLoss",
    "margin": 10.0
})
# Load and preprocess the data
image_paths = []
labels = []

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for image_name in os.listdir(label_dir):
            image_paths.append(os.path.join(label_dir, image_name))
            labels.append(label)

# Debugging output
print(f"Found {len(image_paths)} images in {len(set(labels))} classes")

# Split the data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Check if the splits are non-empty
print(f"Training on {len(train_paths)} images")
print(f"Validating on {len(val_paths)} images")

# Create data generators
num_classes = len(set(labels))
train_generator = TripletDataGenerator(train_paths, train_labels, batch_size, image_size, num_classes)
val_generator = TripletDataGenerator(val_paths, val_labels, batch_size, image_size, num_classes)

# Check if the generators have data
assert len(train_generator) > 0, "Training generator is empty!"
assert len(val_generator) > 0, "Validation generator is empty!"

# Create the embedding model and the Siamese network
embedding_model = get_embedding_module(image_size)
siamese_network = get_siamese_network(image_size, embedding_model)

# Initialize the Siamese model
loss_tracker = Mean(name="loss")
siamese_model = SiameseModel(siamese_network, margin, loss_tracker)

# Compile the model
siamese_model.compile(optimizer=Adam())

# Train the model
siamese_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[WandbMetricsLogger(log_freq=5)] # Remove options and add save_weights_only
)