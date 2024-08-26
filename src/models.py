import tensorflow as tf
import numpy as np
class MultiOutputModel(tf.keras.Model):
    def __init__(self, models):
        super(MultiOutputModel, self).__init__()
        self.models = models 

    def predict(self, inputs, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        # Override the predict method
        return np.vstack([np.argmax(model.predict(inputs)[:,6:], axis=1) for model in self.models]).T