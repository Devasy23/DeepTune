import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger
import wandb

from deeptuner.backbones import ResNetBackbone, EfficientNetBackbone, MobileNetBackbone
from deeptuner.architectures.siamese import SiameseArchitecture
from deeptuner.losses import triplet_loss, arcface_loss, contrastive_loss, center_loss, npair_loss
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
backbone_name = config['backbone']
loss_function_name = config['loss_function']
fine_tune_learning_rate = config['fine_tune_learning_rate']
scale = config['scale']
arcface_margin = config['arcface_margin']

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
if backbone_name == "resnet":
    backbone = ResNetBackbone(input_shape=image_size + (3,))
elif backbone_name == "efficientnet":
    backbone = EfficientNetBackbone(input_shape=image_size + (3,))
elif backbone_name == "mobilenet":
    backbone = MobileNetBackbone(input_shape=image_size + (3,))
else:
    raise ValueError(f"Unsupported backbone: {backbone_name}")

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

# Select the loss function
if loss_function_name == "triplet_loss":
    loss_function = triplet_loss(margin=margin)
elif loss_function_name == "arcface_loss":
    loss_function = arcface_loss(scale=scale, margin=arcface_margin)
elif loss_function_name == "contrastive_loss":
    loss_function = contrastive_loss(margin=margin)
elif loss_function_name == "center_loss":
    loss_function = center_loss()
elif loss_function_name == "npair_loss":
    loss_function = npair_loss()
else:
    raise ValueError(f"Unsupported loss function: {loss_function_name}")

# Compile the model
siamese_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function)

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

# Function to load LFW dataset
def load_lfw_dataset(data_dir):
    image_paths = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_paths.append(os.path.join(label_dir, image_name))
                labels.append(label)

    return image_paths, labels

# Example usage
if __name__ == "__main__":
    data_dir = "path/to/your/dataset"
    image_paths, labels = load_lfw_dataset(data_dir)
    print(f"Loaded {len(image_paths)} images from LFW dataset")
