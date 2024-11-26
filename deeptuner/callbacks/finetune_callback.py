import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

class FineTuneCallback(Callback):
    def __init__(self, base_model, patience=5, unfreeze_layers=10, margin=1.0):
        super(FineTuneCallback, self).__init__()
        self.base_model = base_model
        self.patience = patience
        self.unfreeze_layers = unfreeze_layers
        self.margin = margin
        self.best_weights = None
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Restore the best weights
                self.model.set_weights(self.best_weights)
                self.wait = 0
                # Unfreeze the last few layers
                for layer in self.base_model.layers[-self.unfreeze_layers:]:
                    if hasattr(layer, 'trainable'):
                        layer.trainable = True
                # Recompile the model with both optimizer and loss function
                from deeptuner.losses.triplet_loss import triplet_loss
                self.model.compile(
                    optimizer=Adam(learning_rate=1e-6),
                    loss=triplet_loss(margin=self.margin)
                )
