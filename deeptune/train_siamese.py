import os
import wandb
import tensorflow as tf
from .models.base import SiameseBase
from .models.backbones import BackboneFactory
from .losses import LossFactory
from .data import load_image_dataset, TripletGenerator

# Initialize wandb
wandb.init(
    project="DeepTune-Example",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "image_size": [224, 224],
        "backbone": "resnet50",
        "loss": "triplet",
        "margin": 0.6,
        "embedding_dim": 128
    }
)
config = wandb.config

def main():
    # Model parameters
    input_shape = tuple(config.image_size + [3])
    embedding_dim = config.embedding_dim
    margin = config.margin
    batch_size = config.batch_size
    epochs = config.epochs
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "lfw_processed")
    train_paths, train_labels, val_paths, val_labels = load_image_dataset(
        data_dir,
        config.image_size,
        test_split=0.2
    )

    # Create backbone network
    backbone_factory = BackboneFactory()
    backbone = backbone_factory.create(
        config.backbone,
        input_shape=input_shape,
        embedding_dim=embedding_dim
    )
    
    # Create loss function
    loss_factory = LossFactory()
    loss_fn = loss_factory.create(config.loss, margin=margin)
    
    # Create Siamese model
    siamese_model = SiameseBase(
        embedding_network=backbone,
        loss_fn=loss_fn
    )
    
    # Compile model
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate)
    )
    
    # Create data generators
    train_generator = TripletGenerator(
        train_paths,
        train_labels,
        batch_size,
        config.image_size
    )
    val_generator = TripletGenerator(
        val_paths,
        val_labels,
        batch_size,
        config.image_size
    )
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train model
    siamese_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[
            wandb.keras.WandbCallback(),
            tf.keras.callbacks.ModelCheckpoint(
                "checkpoints/model_{epoch:02d}.keras",
                save_best_only=True,
                monitor="loss"
            )
        ]
    )

    # Save model
    os.makedirs("models", exist_ok=True)
    siamese_model.save("models/siamese_model.keras")

if __name__ == "__main__":
    main()
