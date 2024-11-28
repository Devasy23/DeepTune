import os
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger
import wandb
import shutil
from pathlib import Path

from deeptuner.backbones.resnet import ResNetBackbone
from deeptuner.losses.arcface_loss import ArcFaceModel, arcface_loss, ArcFaceLayer
from deeptuner.datagenerators.arcface_generator import ArcFaceDataGenerator

config = {
    "data_dir": "/kaggle/input/indian-actor-faces-for-face-recognition/actors_dataset/Indian_actors_faces",
    "image_size": [224, 224],
    "batch_size": 32,
    "epochs": 50,
    "initial_epoch": 0,
    "learning_rate": 0.001,
    "patience": 5,
    "unfreeze_layers": 10,
    "project_name": "DeepTuner",
    "embedding_dim": 512,
    "arcface_margin": 0.5,
    "arcface_scale": 64.0
}

data_dir = config['data_dir']
image_size = tuple(config['image_size'])
batch_size = config['batch_size']
epochs = config['epochs']
initial_epoch = config['initial_epoch']
learning_rate = config['learning_rate']
patience = config['patience']
unfreeze_layers = config['unfreeze_layers']
embedding_dim = config['embedding_dim']
arcface_margin = config['arcface_margin']
arcface_scale = config['arcface_scale']

# Initialize W&B
wandb.init(project=config['project_name'], config=config)

# Create train and validation directories
train_dir = Path('temp_train')
val_dir = Path('temp_val')

# Clean up any existing temp directories
if train_dir.exists():
    shutil.rmtree(train_dir)
if val_dir.exists():
    shutil.rmtree(val_dir)

train_dir.mkdir(parents=True)
val_dir.mkdir(parents=True)

# Load and split the data
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
num_classes = len(classes)

print(f"Found {num_classes} classes")

# Split data and create temporary directory structure
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split images into train and validation
    train_images, val_images = train_test_split(
        images, test_size=0.2, random_state=42
    )
    
    # Create class directories in train and val
    train_class_dir = train_dir / class_name
    val_class_dir = val_dir / class_name
    train_class_dir.mkdir(exist_ok=True)
    val_class_dir.mkdir(exist_ok=True)
    
    # Copy images to respective directories
    for img in train_images:
        shutil.copy2(
            os.path.join(class_dir, img),
            train_class_dir / img
        )
    for img in val_images:
        shutil.copy2(
            os.path.join(class_dir, img),
            val_class_dir / img
        )

print(f"Data split complete. Training directory: {train_dir}, Validation directory: {val_dir}")

# Create data generators
train_generator = ArcFaceDataGenerator(
    data_dir=str(train_dir),
    batch_size=batch_size,
    image_size=image_size,
    augment=True,
    cache=True
)

val_generator = ArcFaceDataGenerator(
    data_dir=str(val_dir),
    batch_size=batch_size,
    image_size=image_size,
    augment=False,
    cache=True
)

# Create datasets
train_dataset = train_generator.create_dataset(is_training=True)
val_dataset = val_generator.create_dataset(is_training=False)

# Create the backbone model
backbone = ResNetBackbone(input_shape=image_size + (3,))
backbone_model = backbone.create_model()

# Freeze initial layers
for layer in backbone_model.layers[:-unfreeze_layers]:
    layer.trainable = False

# Create ArcFace model
model = ArcFaceModel(
    backbone=backbone_model,
    num_classes=num_classes,
    embedding_dim=embedding_dim,
    margin=arcface_margin,
    scale=arcface_scale
)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=arcface_loss(),
    metrics=['accuracy']
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Set up callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'models/best_arcface_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    WandbMetricsLogger(log_freq=5)
]

# Training
print("Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    steps_per_epoch=train_generator.steps_per_epoch
)

# Save the final model
model.save('models/final_arcface_model.keras')

# Clean up temporary directories
shutil.rmtree(train_dir)
shutil.rmtree(val_dir)

print("Training completed!")