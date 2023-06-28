import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight
from data_prep import train, test


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Utils
def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls='dashed')
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history['val_' + metric]);
        ax[idx].legend([metric, 'val_' + metric])

# Custom metrics
class pFBeta(tf.keras.metrics.Metric):
    def __init__(self, beta=1, name='pF1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.epsilon = 1e-10
        self.pos = self.add_weight(name='pos', initializer='zeros')
        self.ctp = self.add_weight(name='ctp', initializer='zeros')
        self.cfp = self.add_weight(name='cfp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        pos = tf.cast(tf.reduce_sum(y_true), tf.float32)
        ctp = tf.cast(tf.reduce_sum(y_pred[y_true == 1]), tf.float32)
        cfp = tf.cast(tf.reduce_sum(y_pred[y_true == 0]), tf.float32)
        self.pos.assign_add(pos)
        self.ctp.assign_add(ctp)
        self.cfp.assign_add(cfp)

    def result(self):
        beta2 = self.beta * self.beta
        prec = self.ctp / (self.ctp + self.cfp + self.epsilon)
        reca = self.ctp / (self.pos + self.epsilon)
        return (1 + beta2) * prec * reca / (beta2 * prec + reca)

    def reset_state(self):
        self.pos.assign(0.)
        self.ctp.assign(0.)
        self.cfp.assign(0.)

# Calculate class weights 
def class_weight(labels):
    train_labels = labels.loc[:, '0']

    class_weights = compute_class_weight(class_weight = "balanced",
                                     classes= np.unique(train_labels),
                                     y= train_labels)

    class_weights = dict(zip(np.unique(train_labels), class_weights))
    return class_weights

def build_model(num_hidden = 32, lrate=0.001, l1 = 0. ,
                 l2 = 0., num_classes=1):
    regularizer = tf.keras.regularizers.l1_l2(l1, l2)
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet",
                                                                   input_shape= (256, 256, 3),
                                                                   pooling= 'max')
    model = tf.keras.models.Sequential([
        base_model,
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')])
    return model

metrics = [pFBeta(beta=1, name='pF1'),
                   tfa.metrics.F1Score(num_classes=1, threshold=0.50, name='F1'),
                   tf.metrics.Precision(name='Prec'),
                   tf.metrics.Recall(name='Reca'),
                   tf.metrics.AUC(name='AUC'),
                   tf.metrics.BinaryAccuracy(name='BinAcc')]

def train_model(model):
    history = model.fit(train, 
    epochs=10,
    batch_size=32,
    validation_data=test,
    class_weight={0: 0.5107791269489826,
                   1: 23.692972972972974})
    
    return history

if __name__ == "__main__":
    model = build_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.experimental.SGD(momentum=0.9),
                loss= tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.1), metrics= metrics)
    print("Model compiled correctly...")
    print("Initializing training...")
    history = train_model(model)

