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

    data_dir: str = ""
    img_width: int = 512
    image_height: int = 512


    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    X_data = generate_np_array_from_batch_image("")

    Y_label_file = None
    Y_label = np.load(Y_label_file)

    # (18000, 150, 150)
    # (18000, 2)


    epoch = 20
    # train_size_list = [0.2, 0.4]
    train_size_list = [0.2, 0.4, 0.6, 0.8]

    X_data = X_data.reshape([-1, img_width, image_height, 1])/255


    # set up cnn model
    inputs = Input(shape=(150,150,1))
    shared_layer = Conv2D(32, (3, 3), strides=2, activation='relu')(inputs)
    shared_layer = MaxPooling2D((2, 2))(shared_layer)
    shared_layer = Conv2D(64, (3, 3), strides=1, activation='relu')(shared_layer)
    shared_layer = MaxPooling2D((2, 2))(shared_layer)
    shared_layer = Conv2D(64, (3, 3), strides=1, activation='relu')(shared_layer)
    shared_layer = Flatten()(shared_layer)


    hour_layer = Dense(64, activation='relu')(shared_layer)
    hour_layer = Dense(4, activation='softmax', name='specimen_type')(hour_layer)

    cls_model = Model(inputs=inputs, outputs=[hour_layer])
    cls_model.compile(loss=['CategoricalCrossentropy'], optimizer='rmsprop', metrics=['accuracy'], run_eagerly=True)


    # execute data set
    classification_log_list = []

    for train_size in train_size_list:
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_label, train_size=train_size)


        history = cls_model.fit(X_train, Y_train, epochs=epoch)

        Y_train_softmax_pred = cls_model.predict(X_test)

        Y_hour_train_softmax_pred = np.array(Y_train_softmax_pred[0])
        Y_minutes_train_softmax_pred = np.array(Y_train_softmax_pred[1])
        Y_train_hr_pred = np.argmax(Y_hour_train_softmax_pred, axis=1)
        Y_train_minute_pred = np.argmax(Y_minutes_train_softmax_pred, axis=1)
        Y_train_hr_minute_pred = [Y_train_hr_pred[i]*60 + Y_train_minute_pred[i] for i in range(len(Y_train_hr_pred)) ]

        Y_train_hr_accuracy = round(accuracy_score(Y_train_hour_label, Y_train_hr_pred), 4)
        Y_train_minute_accuracy = round(accuracy_score(Y_train_minute_label, Y_train_minute_pred), 4)
        Y_train_hr_minute_accuracy = round(accuracy_score(Y_train_hr_minute_label, Y_train_hr_minute_pred), 4)
        train_minute_loss = derive_minute_loss_from_classification(Y_train_hr_minute_label, Y_train_hr_minute_pred)


        Y_test_softmax_pred = cls_model.predict(X_test)

        Y_hour_test_softmax_pred = np.array(Y_test_softmax_pred[0])
        Y_minutes_test_softmax_pred = np.array(Y_test_softmax_pred[1])
        Y_test_hr_pred = np.argmax(Y_hour_test_softmax_pred, axis=1)
        Y_test_minute_pred = np.argmax(Y_minutes_test_softmax_pred, axis=1)
        Y_test_hr_minute_pred = [Y_test_hr_pred[i]*60 + Y_test_minute_pred[i] for i in range(len(Y_test_hr_pred)) ]

        Y_test_hr_accuracy = round(accuracy_score(Y_test_hour_label, Y_test_hr_pred), 4)
        Y_test_minute_accuracy = round(accuracy_score(Y_test_minute_label, Y_test_minute_pred), 4)
        Y_test_hr_minute_accuracy = round(accuracy_score(Y_test_hr_minute_label, Y_test_hr_minute_pred), 4)
        test_minute_loss = derive_minute_loss_from_classification(Y_test_hr_minute_label, Y_test_hr_minute_pred)


        log_msg = f"train_size: {train_size*100}%; Y_train_hr_minute_accuracy: {Y_train_hr_minute_accuracy}; Y_train_hr_accuracy: {Y_train_hr_accuracy}; Y_train_minute_accuracy: {Y_train_minute_accuracy}; train_minute_loss: {train_minute_loss}; Y_test_hr_minute_accuracy: {Y_test_hr_minute_accuracy}; Y_test_hr_accuracy: {Y_test_hr_accuracy}; Y_test_minute_accuracy: {Y_test_minute_accuracy}; test_minute_loss: {test_minute_loss}; "
        classification_log_list.append(log_msg)
        print(log_msg)

        test_hour_cf = confusion_matrix(Y_test_hour_label, Y_test_hr_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=test_hour_cf)
        disp.plot()



def generate_np_array_from_batch_image(image_dir)
    image_file_list = os.listdir(image_dir)
    return ""

if __name__ == '__main__':
    main()