import tensorflow as tf

class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name="triplet_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        """Compute triplet loss.
        Args:
            y_true: Not used in triplet loss
            y_pred: Stacked embeddings of shape (batch_size, 3, embedding_dim)
                   where y_pred[:, 0] = anchor
                         y_pred[:, 1] = positive
                         y_pred[:, 2] = negative
        """
        # Unpack the stacked embeddings
        anchor, positive, negative = tf.unstack(y_pred, axis=1)
        
        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calculate triplet loss
        basic_loss = pos_dist - neg_dist + self.margin
        loss = tf.maximum(basic_loss, 0.0)
        
        return tf.reduce_mean(loss)

def arcface_loss(embeddings, y_true, margin=0.5, scale=64):
    """Compute ArcFace loss.
    
    Args:
        embeddings: Feature embeddings
        y_true: True labels
        margin: Angular margin
        scale: Scale factor
    """
    # Normalize embeddings
    embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
    
    # Convert labels to one-hot
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, depth=embeddings.shape[-1])
    
    # Calculate cosine similarity
    cos_t = tf.matmul(embeddings_norm, tf.transpose(embeddings_norm))
    
    # Add angular margin
    theta = tf.acos(cos_t)
    marginal_target = tf.cos(theta + margin)
    
    # Scale and combine
    cos_t = tf.where(y_true > 0, marginal_target, cos_t)
    output = scale * cos_t
    
    # Calculate cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true, output)
    return tf.reduce_mean(loss)

class LossFactory:
    """Factory class for creating loss functions."""
    
    @staticmethod
    def create(loss_type, **kwargs):
        """Create a loss function.
        
        Args:
            loss_type: Type of the loss function (triplet, arcface)
            **kwargs: Additional arguments for the loss function
        """
        if loss_type.lower() == "triplet":
            return TripletLoss(**kwargs)
        elif loss_type.lower() == "arcface":
            def loss_wrapper(y_true, y_pred, margin=kwargs.get("margin", 0.5), scale=kwargs.get("scale", 64)):
                return arcface_loss(y_pred, y_true, margin=margin, scale=scale)
            return loss_wrapper
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
