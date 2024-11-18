import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import resnet, efficientnet, inception_resnet_v2

class BackboneFactory:
    """Factory class for creating backbone networks."""
    
    @staticmethod
    def create(name, input_shape, embedding_dim=128, trainable=False, **kwargs):
        """Create a backbone network.
        
        Args:
            name: Name of the backbone (resnet50, efficientnet, inceptionresnetv1)
            input_shape: Input shape of the images
            embedding_dim: Dimension of the output embedding
            trainable: Whether to train the backbone
            **kwargs: Additional arguments for the backbone
        """
        backbone_map = {
            "resnet50": BackboneFactory._create_resnet50,
            "efficientnet": BackboneFactory._create_efficientnet,
            "inceptionresnetv1": BackboneFactory._create_inception_resnet
        }
        
        if name not in backbone_map:
            raise ValueError(f"Unsupported backbone: {name}")
            
        return backbone_map[name](input_shape, embedding_dim, trainable, **kwargs)
    
    @staticmethod
    def _create_resnet50(input_shape, embedding_dim, trainable, **kwargs):
        inputs = tf.keras.Input(input_shape)
        x = resnet.preprocess_input(inputs)
        
        base_model = resnet.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = trainable
        
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(embedding_dim)(x)
        
        return tf.keras.Model(inputs, outputs, name="resnet50_backbone")
    
    @staticmethod
    def _create_efficientnet(input_shape, embedding_dim, trainable, **kwargs):
        inputs = tf.keras.Input(input_shape)
        x = efficientnet.preprocess_input(inputs)
        
        base_model = efficientnet.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = trainable
        
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(embedding_dim)(x)
        
        return tf.keras.Model(inputs, outputs, name="efficientnet_backbone")
    
    @staticmethod
    def _create_inception_resnet(input_shape, embedding_dim, trainable, **kwargs):
        inputs = tf.keras.Input(input_shape)
        x = inception_resnet_v2.preprocess_input(inputs)
        
        base_model = inception_resnet_v2.InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = trainable
        
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(embedding_dim)(x)
        
        return tf.keras.Model(inputs, outputs, name="inception_resnet_backbone")
