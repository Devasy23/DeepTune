import tensorflow as tf
from tensorflow.keras import backend as K

def arcface_loss(scale=64.0, margin=0.5):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.one_hot(y_true, depth=y_pred.shape[-1])
        
        cos_theta = y_pred
        theta = tf.acos(K.clip(cos_theta, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + margin)
        
        logits = y_true * target_logits + (1 - y_true) * cos_theta
        logits *= scale
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        return tf.reduce_mean(loss)
    
    return loss
