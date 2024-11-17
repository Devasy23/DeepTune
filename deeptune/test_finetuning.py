import tensorflow as tf
from DeepTune import triplet_loss, backbones, data_preprocessing, evaluation_metrics, datasets

def test_finetuning():
    # Load and preprocess data
    train_triplets, test_triplets = data_preprocessing.preprocess_data('path/to/dataset')

    # Initialize model backbone
    model = backbones.get_model('resnet')

    # Compile model with triplet loss
    model.compile(optimizer='adam', loss=triplet_loss.triplet_loss)

    # Train model
    model.fit(train_triplets, epochs=5, batch_size=32)

    # Evaluate model
    metrics = evaluation_metrics.evaluate_model(model, test_triplets)
    print(metrics)

if __name__ == "__main__":
    test_finetuning()
