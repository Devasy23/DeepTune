import tensorflow as tf
import math
from tensorflow.keras import backend as K


class ArcFaceLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, embedding_dim=512, margin=0.5, scale=64.0, regularizer_l=5e-4, **kwargs):
        super(ArcFaceLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.regularizer_l = regularizer_l
        
        # Pre-calculate angular values
        self.cos_m = tf.cast(math.cos(margin), dtype=tf.float32)
        self.sin_m = tf.cast(math.sin(margin), dtype=tf.float32)
        self.th = tf.cast(math.cos(math.pi - margin), dtype=tf.float32)
        self.mm = tf.cast(math.sin(math.pi - margin) * margin, dtype=tf.float32)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(self.embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.regularizer_l),
            trainable=True
        )
        
    def call(self, inputs, labels):
        # Normalize features and weights
        x = tf.nn.l2_normalize(inputs, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        
        # Calculate cosine similarity
        cos_t = tf.matmul(x, W)
        
        # Clip for numerical stability
        cos_t = tf.clip_by_value(cos_t, -1.0 + K.epsilon(), 1.0 - K.epsilon())
        
        # Calculate required angles
        sin_t = tf.sqrt(1.0 - tf.pow(cos_t, 2))
        cos_mt = tf.multiply(cos_t, self.cos_m) - tf.multiply(sin_t, self.sin_m)
        
        # Safe zone: choose cosine when cos(theta) <= cos(pi - margin)
        cond_v = cos_t - self.th
        cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
        keep_val = self.scale * (cos_t - self.mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        
        # Convert labels to one-hot
        mask = tf.one_hot(tf.cast(labels, dtype=tf.int32), depth=self.num_classes)
        
        # Apply margins
        logits = tf.where(mask == 1., cos_mt_temp, cos_t)
        logits = logits * self.scale
        
        return logits


def arcface_loss():
    def loss(y_true, y_pred):
        # y_true in this case should be the original labels
        # y_pred will be the output from ArcFaceLayer (already scaled logits)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(y_true, dtype=tf.int32),
                logits=y_pred
            )
        )
    return loss


class ArcFaceModel(tf.keras.Model):
    def __init__(self, backbone, num_classes, embedding_dim=512, margin=0.5, scale=64.0):
        super(ArcFaceModel, self).__init__()
        self.backbone = backbone
        self.embedding_layer = tf.keras.layers.Dense(embedding_dim)
        self.bn = tf.keras.layers.BatchNormalization()
        self.arcface = ArcFaceLayer(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=margin,
            scale=scale
        )
        
    def call(self, inputs, training=False):
        features = self.backbone(inputs, training=training)
        embeddings = self.embedding_layer(features)
        embeddings = self.bn(embeddings, training=training)
        return embeddings
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            embeddings = self(x, training=True)
            logits = self.arcface(embeddings, y)
            loss = self.compiled_loss(y, logits)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, logits)
        
        return {m.name: m.result() for m in self.metrics}
