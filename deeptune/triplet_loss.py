import tensorflow as tf

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Custom triplet loss function.

    Args:
        anchor: The anchor input tensor.
        positive: The positive input tensor.
        negative: The negative input tensor.
        margin: The margin by which the positive example should be closer to the anchor than the negative example.

    Returns:
        The triplet loss value.
    """
    # Calculate the distance between the anchor and the positive example
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)

    # Calculate the distance between the anchor and the negative example
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    # Compute the triplet loss
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)

    return tf.reduce_mean(loss)
