# DeepTuner

## Description

DeepTuner is an open source Python package for fine-tuning computer vision (CV) based deep models using Siamese architecture with a triplet loss function. The package supports various model backbones and provides tools for data preprocessing and evaluation metrics.

## Installation

To install the package, use the following command:

```bash
pip install DeepTuner
```

## Usage

### Fine-tuning Models with Siamese Architecture and Triplet Loss

Here is an example of how to use the package for fine-tuning models with Siamese architecture and triplet loss:

```python
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger
import wandb

from deeptuner.backbones.resnet import ResNetBackbone
from deeptuner.architectures.siamese import SiameseArchitecture
from deeptuner.losses.triplet_loss import triplet_loss
from deeptuner.datagenerators.triplet_data_generator import TripletDataGenerator
from deeptuner.callbacks.finetune_callback import FineTuneCallback

# Load configuration from JSON file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

data_dir = config['data_dir']
image_size = tuple(config['image_size'])
batch_size = config['batch_size']
margin = config['margin']
epochs = config['epochs']
initial_epoch = config['initial_epoch']
learning_rate = config['learning_rate']
patience = config['patience']
unfreeze_layers = config['unfreeze_layers']

# Initialize W&B
wandb.init(project=config['project_name'], config=config)

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

# Create the embedding model and freeze layers
backbone = ResNetBackbone(input_shape=image_size + (3,))
embedding_model = backbone.create_model()

# Freeze all layers initially
for layer in embedding_model.layers:
    layer.trainable = False
# Unfreeze last few layers
for layer in embedding_model.layers[-unfreeze_layers:]:
    layer.trainable = True

# Create the siamese network
siamese_architecture = SiameseArchitecture(input_shape=image_size + (3,), embedding_model=embedding_model)
siamese_network = siamese_architecture.create_siamese_network()

# Initialize the Siamese model
loss_tracker = Mean(name="loss")
siamese_model = SiameseModel(siamese_network, margin, loss_tracker)

# Set up callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(
    "models/best_siamese_model.weights.h5", 
    save_best_only=True, 
    save_weights_only=True, 
    monitor='val_loss', 
    verbose=1
)
embedding_checkpoint = ModelCheckpoint(
    "models/best_embedding_model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    verbose=1
)
fine_tune_callback = FineTuneCallback(embedding_model, patience=patience, unfreeze_layers=unfreeze_layers)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=triplet_loss(margin=margin))

# Train the model
history = siamese_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=[
        reduce_lr, 
        early_stopping, 
        model_checkpoint,
        embedding_checkpoint,
        fine_tune_callback,
        WandbMetricsLogger(log_freq=5)
    ]
)

# Save the final embedding model
embedding_model.save('models/final_embedding_model.h5')
```

### Using Configuration Files

To make it easier to experiment with different hyperparameter settings, you can use a configuration file (e.g., JSON) to store hyperparameters. Here is an example of a configuration file (`config.json`):

```json
{
    "data_dir": "path/to/your/dataset",
    "image_size": [224, 224],
    "batch_size": 32,
    "margin": 1.0,
    "epochs": 50,
    "initial_epoch": 0,
    "learning_rate": 0.001,
    "patience": 5,
    "unfreeze_layers": 10,
    "project_name": "DeepTuner"
}
```

You can then load this configuration file in your code as shown in the usage example above.
