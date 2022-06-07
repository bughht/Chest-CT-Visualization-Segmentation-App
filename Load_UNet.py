'''
Author: Harryhht
Date: 2022-02-22 08:13:52
LastEditors: Harryhht
LastEditTime: 2022-02-22 19:12:13
Description:
'''
import os
import cv2
import random
import pydicom
import warnings
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import get_custom_objects

warnings.filterwarnings('ignore')
print('Tensorflow version : {}'.format(tf.__version__))


def get_segmentation_model():

    class FixedDropout(tf.keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = tf.keras.backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    def DiceCoef(y_trues, y_preds, smooth=1e-5, axis=None):
        intersection = tf.reduce_sum(y_trues * y_preds, axis=axis)
        union = tf.reduce_sum(y_trues, axis=axis) + \
            tf.reduce_sum(y_preds, axis=axis)
        return tf.reduce_mean((2*intersection+smooth) / (union + smooth))

    def DiceLoss(y_trues, y_preds):
        return 1.0 - DiceCoef(y_trues, y_preds)

    get_custom_objects().update(
        {'swish': tf.keras.layers.Activation(tf.nn.swish)})
    get_custom_objects().update({'FixedDropout': FixedDropout})
    get_custom_objects().update({'DiceCoef': DiceCoef})
    get_custom_objects().update({'DiceLoss': DiceLoss})
    print('Load segmentation model...')
    model = tf.keras.models.load_model('ct/osic_segmentation_model.h5')
    return model


model = get_segmentation_model()
print(model.summary())

DEMO_BATCH = 8
for idx in range(DEMO_BATCH):
    images =
    pred_masks = model.predict(images, verbose=0)
    pred_masks = (pred_masks > 0.5).astype(np.float32)

    plt.figure(figsize=(24, 12))
    for idx, (image, mask) in enumerate(zip(images, pred_masks)):
        plt.subplot(1, 8, idx+1)
        plt.imshow(image)
        plt.imshow(mask[:, :, 0], alpha=0.35)
        plt.xticks([])
        plt.yticks([])
    plt.show()
