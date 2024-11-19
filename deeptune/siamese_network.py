from tensorflow import keras
import tensorflow as tf

class SiameseModel(keras.Model):
    def __init__(self, siameseNetwork, margin, lossTracker):
        super().__init__()
        self.siameseNetwork = siameseNetwork
        self.margin = margin
        self.lossTracker = lossTracker

    def _compute_distance(self, inputs):
        anchor, positive, negative = inputs["anchor"], inputs["positive"], inputs["negative"]
        # embed the images using the siamese network
        embeddings = self.siameseNetwork((anchor, positive, negative))
        anchorEmbedding = embeddings[0]
        positiveEmbedding = embeddings[1]
        negativeEmbedding = embeddings[2]
        # calculate the anchor to positive and negative distance
        apDistance = tf.reduce_sum(
            tf.square(anchorEmbedding - positiveEmbedding), axis=-1
        )
        anDistance = tf.reduce_sum(
            tf.square(anchorEmbedding - negativeEmbedding), axis=-1
        )
        # return the distances
        return (apDistance, anDistance)

    def _compute_loss(self, apDistance, anDistance):
        loss = apDistance - anDistance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def call(self, inputs):
        # compute the distance between the anchor and positive,
        # negative images
        apDistance, anDistance = self._compute_distance(inputs)
        return (apDistance, anDistance)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        
        with tf.GradientTape() as tape:
            # compute the distance between the anchor and positive,
            # negative images
            apDistance, anDistance = self._compute_distance(x)
            # calculate the loss of the siamese network
            loss = self._compute_loss(apDistance, anDistance)
            
        # compute the gradients and optimize the model
        gradients = tape.gradient(
            loss,
            self.siameseNetwork.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.siameseNetwork.trainable_variables)
        )
        # update the metrics and return the loss
        self.lossTracker.update_state(loss)
        return {"loss": self.lossTracker.result()}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # compute the distance between the anchor and positive,
        # negative images
        apDistance, anDistance = self._compute_distance(x)
        # calculate the loss of the siamese network
        loss = self._compute_loss(apDistance, anDistance)
        
        # update the metrics and return the loss
        self.lossTracker.update_state(loss)
        return {"loss": self.lossTracker.result()}

    @property
    def metrics(self):
        return [self.lossTracker]

    def get_config(self):
        config = super().get_config()
        config.update({
            "siameseNetwork": self.siameseNetwork,
            "margin": self.margin,
            "lossTracker": self.lossTracker
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
