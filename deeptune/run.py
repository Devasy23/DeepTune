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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
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

class FineTuneCallback(keras.callbacks.Callback):
    def __init__(self, base_model, patience=5, unfreeze_layers=10):
        super(FineTuneCallback, self).__init__()
        self.base_model = base_model
        self.patience = patience
        self.unfreeze_layers = unfreeze_layers
        self.best_weights = None
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Restore the best weights
                self.model.set_weights(self.best_weights)
                self.wait = 0
                # Unfreeze the last few layers
                for layer in self.base_model.layers[-self.unfreeze_layers:]:
                    if hasattr(layer, 'trainable'):
                        layer.trainable = True
                # Recompile the model to apply the changes
                self.model.compile(optimizer=Adam(learning_rate=1e-5))

# Create the embedding model and freeze layers
embedding_model = get_embedding_module(image_size)
# Freeze all layers initially
for layer in embedding_model.layers:
    layer.trainable = False
# Unfreeze last 20 layers
for layer in embedding_model.layers[-20:]:
    layer.trainable = True

# Create the siamese network
siamese_network = get_siamese_network(image_size, embedding_model)

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
fine_tune_callback = FineTuneCallback(embedding_model, patience=5, unfreeze_layers=10)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Compile the model
siamese_model.compile(optimizer=Adam())

# Train the model
history = siamese_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    initial_epoch=20,
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