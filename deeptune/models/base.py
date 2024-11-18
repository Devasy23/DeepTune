from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras

class BaseModel(ABC, keras.Model):
    """Base class for all models in DeepTune."""
    
    def __init__(self, name="base_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
    @abstractmethod
    def call(self, inputs):
        """Forward pass of the model."""
        pass
    
    @property
    def metrics(self):
        """Model metrics."""
        return [self.loss_tracker]

class SiameseBase(tf.keras.Model):
    """Base class for Siamese networks."""
    
    def __init__(self, embedding_network, loss_fn, name="siamese_base", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_network = embedding_network
        self.loss_fn = loss_fn
        # Metric to track loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def call(self, inputs):
        """Forward pass of the model."""
        if isinstance(inputs, dict):
            # During training, expect dictionary with anchor, positive, negative
            embeddings = self._compute_embeddings(inputs)
            return embeddings
        else:
            # During inference, expect single input
            return self.embedding_network(inputs)
    
    def _compute_embeddings(self, inputs):
        """Compute embeddings for anchor, positive, and negative inputs."""
        anchor, positive, negative = inputs["anchor"], inputs["positive"], inputs["negative"]
        anchor_embedding = self.embedding_network(anchor)
        positive_embedding = self.embedding_network(positive)
        negative_embedding = self.embedding_network(negative)
        
        # Stack embeddings for loss computation
        return tf.stack([anchor_embedding, positive_embedding, negative_embedding], axis=1)
    
    def train_step(self, data):
        """Training step."""
        inputs, _ = data
        
        with tf.GradientTape() as tape:
            # Get embeddings
            embeddings = self._compute_embeddings(inputs)
            
            # Compute loss - pass dummy y_true since it's not used
            y_true = tf.zeros(tf.shape(embeddings)[0])  # Dummy labels
            loss = self.loss_fn(y_true, embeddings)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
