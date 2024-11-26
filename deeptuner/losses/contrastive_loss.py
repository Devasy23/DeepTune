import tensorflow as tf

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        anchor, positive = y_pred[0], y_pred[1]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        loss = tf.reduce_mean(tf.maximum(pos_dist - margin, 0.0), axis=0)
        return loss
    return loss
