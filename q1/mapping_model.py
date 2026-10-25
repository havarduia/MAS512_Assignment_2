# ML model that learns the parameters theta = (W, b) of f_theta(o1) = W o1 + b = o2*
# As in the ML lecture: linear regression = one-layer perceptron without activation, trained with Adam on MSE loss

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TensorFlow's C++ start-up logs (must be set before import)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # one Dense layer trains fine on the CPU

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MappingModel:
    def __init__(self, train_fraction=0.8, val_fraction=0.1, epochs=500, learning_rate=0.01):
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, o1, o2):
        # split in time order (80 / 10 / 10): neighbouring frames are near-copies, a random split would leak them
        n = len(o1)
        self.n_train = int(self.train_fraction * n)
        self.n_val = int(self.val_fraction * n)
        val_end = self.n_train + self.n_val
        o1_train, o2_train = o1[:self.n_train], o2[:self.n_train]
        o1_val, o2_val = o1[self.n_train:val_end], o2[self.n_train:val_end]
        self.o1_test, self.o2_test = o1[val_end:], o2[val_end:]

        tf.keras.utils.set_random_seed(0)
        # inputs have different scales (z ~3 m, x and y ~0.3 m): normalise with the training set's mean and std
        normalization = tf.keras.layers.Normalization()
        normalization.adapt(o1_train)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(3,)),
            normalization,
            tf.keras.layers.Dense(3),  # one layer, no activation: o2* = W o1 + b
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        self.history = self.model.fit(o1_train, o2_train, validation_data=(o1_val, o2_val), epochs=self.epochs,
                                      verbose=0).history

    def predict(self, o1):
        return self.model.predict(o1, verbose=0)

    def parameters(self):
        # W (outputs x inputs) and b for the raw input, with the normalisation folded in
        normalization, dense = self.model.layers
        mean = np.ravel(normalization.mean)
        std = np.sqrt(np.ravel(normalization.variance))
        kernel, bias = dense.get_weights()
        W = (kernel / std[:, None]).T
        return W, bias - W @ mean

    def kpis(self, multioutput="uniform_average"):
        # on the test set, in mm; multioutput="raw_values" gives one value per axis (x, y, z)
        pred = self.predict(self.o1_test)
        return {"RMSE [mm]": 1e3 * np.sqrt(mean_squared_error(self.o2_test, pred, multioutput=multioutput)),
                "MAE [mm]": 1e3 * mean_absolute_error(self.o2_test, pred, multioutput=multioutput),
                "R2": r2_score(self.o2_test, pred, multioutput=multioutput)}
