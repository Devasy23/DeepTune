from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Input

class ResNetBackbone:
    def __init__(self, input_shape=(224, 224, 3), weights='imagenet'):
        self.input_shape = input_shape
        self.weights = weights

    def create_model(self):
        inputs = Input(shape=self.input_shape)
        base_model = ResNet50(weights=self.weights, include_top=False, input_tensor=inputs)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        outputs = Dense(128)(x)
        model = Model(inputs, outputs, name='resnet_backbone')
        return model
