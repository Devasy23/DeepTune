import os
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger
import wandb

from deeptuner.backbones.resnet import ResNetBackbone
from deeptuner.losses.arcface_loss import ArcFaceModel, arcface_loss, ArcFaceLayer

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

# Load and preprocess the data
image_paths = []
labels = []
label_to_id = {}
current_id = 0

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        if label not in label_to_id:
            label_to_id[label] = current_id
            current_id += 1
        for image_name in os.listdir(label_dir):
            image_paths.append(os.path.join(label_dir, image_name))
            labels.append(label_to_id[label])

num_classes = len(label_to_id)
print(f"Found {len(image_paths)} images in {num_classes} classes")

# Split the data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Training on {len(train_paths)} images")
print(f"Validating on {len(val_paths)} images")

# Create data loading pipeline
def parse_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 128.0
    return image, label

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths))
train_dataset = train_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

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
    callbacks=callbacks
)

# Save the final model
model.save('models/final_arcface_model.keras')
print("Training completed!")