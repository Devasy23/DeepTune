from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

def get_embedding_module(imageSize):
    """
    Creates an embedding module based on ResNet50
    """
    # construct the input layer and pass the inputs through a
    # pre-processing layer
    inputs = keras.Input(imageSize + (3,))
    x = resnet.preprocess_input(inputs)

    # fetch the pre-trained resnet 50 model and freeze the weights
    baseCnn = resnet.ResNet50(weights="imagenet", include_top=False)
    baseCnn.trainable=False

    # pass the pre-processed inputs through the base cnn and get the
    # extracted features from the inputs
    extractedFeatures = baseCnn(x)
    # pass the extracted features through a number of trainable layers
    x = layers.GlobalAveragePooling2D()(extractedFeatures)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(units=128)(x)
    # build the embedding model and return it
    embedding = keras.Model(inputs, outputs, name="embedding")
    return embedding

def get_siamese_network(imageSize, embedding_model):
    """
    Creates a siamese network using the provided embedding module
    Args:
        imageSize: tuple of image dimensions (height, width)
        embedding_model: pre-trained embedding model to use
    """
    # build the anchor, positive and negative input layer
    anchorInput = keras.Input(name="anchor", shape=imageSize + (3,))
    positiveInput = keras.Input(name="positive", shape=imageSize + (3,))
    negativeInput = keras.Input(name="negative", shape=imageSize + (3,))
    
    # embed the anchor, positive and negative images
    anchorEmbedding = embedding_model(anchorInput)
    positiveEmbedding = embedding_model(positiveInput)
    negativeEmbedding = embedding_model(negativeInput)
    
    # build the siamese network and return it
    outputs = [anchorEmbedding, positiveEmbedding, negativeEmbedding]
    return keras.Model(
        inputs=[anchorInput, positiveInput, negativeInput],
        outputs=outputs,
        name="siamese_network"
    )