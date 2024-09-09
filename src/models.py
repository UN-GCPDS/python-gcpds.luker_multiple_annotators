import tensorflow as tf
import numpy as np

class MultiOutputModel(tf.keras.Model):
    def __init__(self, models):
        super(MultiOutputModel, self).__init__()
        self.models = models

    def predict(self, inputs, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        sens_predict = []
        ann_rel = []
        for model in self.models:
            pred = model.predict(inputs)
            sens_predict.append(np.argmax(pred[:,6:], axis=1))
            ann_rel.append(pred[:,:6])

        return np.vstack(sens_predict).T, ann_rel