# DeepTuner

## Description

DeepTuner is an open source Python package for fine-tuning computer vision (CV) based deep models. It supports multiple architectures including Siamese Networks with triplet loss and ArcFace with additive angular margin loss. The package provides various model backbones, data preprocessing tools, and evaluation metrics.

## Installation

To install the package, use the following command:

```bash
pip install DeepTuner
```

## Usage

### Training with ArcFace

DeepTuner now supports ArcFace training with additive angular margin loss, which is particularly effective for face recognition tasks:

```python
from deeptuner.backbones.resnet import ResNetBackbone
from deeptuner.losses.arcface_loss import ArcFaceModel, arcface_loss
from deeptuner.datagenerators.arcface_generator import ArcFaceDataGenerator

# Create data generators
train_generator = ArcFaceDataGenerator(
    data_dir='path/to/train/data',
    batch_size=32,
    image_size=(224, 224),
    augment=True
)

# Create backbone and ArcFace model
backbone = ResNetBackbone(input_shape=(224, 224, 3))
backbone_model = backbone.create_model()

model = ArcFaceModel(
    backbone=backbone_model,
    num_classes=num_classes,
    embedding_dim=512,
    margin=0.5,
    scale=64.0
)

# Compile and train
model.compile(
    optimizer='adam',
    loss=arcface_loss(),
    metrics=['accuracy']
)

model.fit(
    train_generator.create_dataset(is_training=True),
    epochs=50
)
```

### Fine-tuning Models with Siamese Architecture and Triplet Loss

For similarity learning tasks, you can use the Siamese architecture with triplet loss:

```python
from deeptuner.backbones.resnet import ResNetBackbone
from deeptuner.architectures.siamese import SiameseArchitecture
from deeptuner.losses.triplet_loss import triplet_loss
from deeptuner.datagenerators.triplet_data_generator import TripletDataGenerator

# Create data generators
train_generator = TripletDataGenerator(
    train_paths, train_labels, 
    batch_size=32,
    image_size=(224, 224),
    num_classes=num_classes
)

# Create Siamese network
backbone = ResNetBackbone(input_shape=(224, 224, 3))
embedding_model = backbone.create_model()
siamese_architecture = SiameseArchitecture(
    input_shape=(224, 224, 3),
    embedding_model=embedding_model
)
siamese_network = siamese_architecture.create_siamese_network()

# Train the model
model.compile(optimizer='adam', loss=triplet_loss(margin=0.5))
model.fit(train_generator, epochs=50)
```

## Features

- Multiple architectures:
  - Siamese Networks with triplet loss
  - ArcFace with additive angular margin loss
- Various backbone models:
  - ResNet
  - EfficientNet
  - MobileNet
- Specialized data generators:
  - TripletDataGenerator for Siamese networks
  - ArcFaceDataGenerator for ArcFace training
- Training utilities:
  - Fine-tuning callbacks
  - Learning rate scheduling
  - Wandb integration for experiment tracking

## Configuration

You can use a configuration file (e.g., JSON) to store hyperparameters. Example config for ArcFace:

```json
{
    "data_dir": "path/to/data",
    "image_size": [224, 224],
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "patience": 5,
    "unfreeze_layers": 10,
    "embedding_dim": 512,
    "arcface_margin": 0.5,
    "arcface_scale": 64.0
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
