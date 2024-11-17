import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, MobileNet, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import backend as K

def get_model(backbone_name, input_shape=(224, 224, 3)):
    """
    Get the model backbone.

    Args:
        backbone_name: The name of the model backbone.
        input_shape: The input shape of the model.

    Returns:
        The model backbone.
    """
    if backbone_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'inception':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'mobilenet':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif backbone_name == 'facenet':
        base_model = InceptionResNetV2(weights=None, include_top=False, input_shape=input_shape)
        base_model.load_weights('path/to/facenet_weights.h5')
    elif backbone_name == 'arcface':
        base_model = InceptionResNetV2(weights=None, include_top=False, input_shape=input_shape)
        base_model.load_weights('path/to/arcface_weights.h5')
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])

    return model
