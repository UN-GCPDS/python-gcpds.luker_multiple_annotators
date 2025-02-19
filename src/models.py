import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model

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

class MultiOutputCCGPMA(tf.keras.Model):
    def __init__(self, models):
        super(MultiOutputCCGPMA, self).__init__()
        self.models = models
    def predict(self, inputs):
        sens_predict = []
        ann_rel = []
        for model in self.models:
            pred = model.compiled_predict_y(inputs)
            sens_predict.append(np.clip(pred[0][:,0], a_min=0, a_max=10))
            ann_rel.append(pred[0][:,1:])
        return np.vstack(sens_predict).T, ann_rel
    
class model_s2fq:
    def __init__(self):
        self.model = self._build_model()
    def _build_model(self):
        inputs = Input(shape=(8,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(5)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = load_model(filepath)
    
    def fit(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_new):
        return self.model.predict(x_new)