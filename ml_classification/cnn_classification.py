import os

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import math


def main():
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    X_data = generate_np_array_from_batch_image("")

    Y_label_file = None
    Y_label = np.load(Y_label_file)


    epoch = 20
    # train_size_list = [0.2, 0.4]
    train_size_list = [0.2, 0.4, 0.6, 0.8]

    X_data = X_data.reshape([-1,150,150,1])/255



def generate_np_array_from_batch_image(image_dir)
    image_file_list = os.listdir(image_dir)
    return ""

if __name__ == '__main__':
    main()