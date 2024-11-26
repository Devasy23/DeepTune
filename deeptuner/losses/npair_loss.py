import tensorflow as tf

def npair_loss():
    def loss(y_true, y_pred):
        anchor, positive, negatives = y_pred[0], y_pred[1], y_pred[2:]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dists = [tf.reduce_sum(tf.square(anchor - neg), axis=-1) for neg in negatives]
        neg_dist = tf.reduce_mean(neg_dists, axis=0)
        basic_loss = pos_dist - neg_dist + 1.0
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), axis=0)
        return loss
    return loss
