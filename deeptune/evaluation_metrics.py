import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model(model, triplets, labels):
    """
    Evaluate the model using various metrics.

    Args:
        model: The trained model.
        triplets: The triplets of images (anchor, positive, negative).
        labels: The true labels of the triplets.

    Returns:
        A dictionary containing the evaluation metrics.
    """
    anchor, positive, negative = triplets
    anchor_embeddings = model.predict(anchor)
    positive_embeddings = model.predict(positive)
    negative_embeddings = model.predict(negative)

    # Calculate distances
    pos_distances = tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=-1)
    neg_distances = tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=-1)

    # Calculate metrics
    accuracy = accuracy_score(labels, pos_distances < neg_distances)
    precision = precision_score(labels, pos_distances < neg_distances)
    recall = recall_score(labels, pos_distances < neg_distances)
    f1 = f1_score(labels, pos_distances < neg_distances)
    roc_auc = roc_auc_score(labels, pos_distances < neg_distances)
    mean_avg_precision = average_precision_score(labels, pos_distances < neg_distances)
    triplet_loss_value = tf.reduce_mean(tf.maximum(pos_distances - neg_distances + 1.0, 0.0))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'mean_avg_precision': mean_avg_precision,
        'triplet_loss': triplet_loss_value
    }
