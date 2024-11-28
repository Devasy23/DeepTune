import tensorflow as tf

def center_loss(alpha=0.5, num_classes=10, feature_dim=2):
    centers = tf.Variable(tf.zeros([num_classes, feature_dim]), trainable=False)
    
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])
        centers_batch = tf.gather(centers, y_true)
        diff = centers_batch - y_pred

        unique_labels, unique_idx, unique_count = tf.unique_with_counts(y_true)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, y_true, diff)
        with tf.control_dependencies([centers_update_op]):
            loss = tf.reduce_mean(tf.square(y_pred - centers_batch))
        return loss
    
    return loss
