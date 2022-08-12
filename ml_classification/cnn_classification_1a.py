import os

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.models import Model, Sequential
from numpy import uint8
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
import os
import tensorflow as tf


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    img_parent_dir: str = "D:/viterbi linkage/dataset/track_classification_images/"
    data_dir: str = ""
    img_width: int = 512
    image_height: int = 512


    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    # X_data = generate_np_array_from_batch_image("")

    Y_label_file = None
    # Y_label = np.load(Y_label_file)


    # (train_images, train_labels), (test_images, test_labels) = obtain_demo_data_set()



    X_data, Y_label = obtain_cell_track_data_set(img_parent_dir)
    X_data = X_data.reshape((105, 512, 512, 1))
    X_data = X_data.astype('int')
    Y_label = Y_label.reshape((Y_label.shape[0]))

    # print(Y_label.shape)
    # print(Y_label[0])
    # exit()


    shuffle_order = np.random.permutation(len(X_data))
    X_data = X_data[shuffle_order]
    Y_label = Y_label[shuffle_order]


    # print(X_data[0][444])
    # print(X_data.shape)
    # exit()

    train_images = X_data[0: 70]
    train_labels = Y_label[0: 70]
    test_images = X_data[70: 104]
    test_labels = Y_label[70: 104]


    # print("dfbsdfn", Y_label)
    # exit()


    # print(train_images.shape, train_labels.shape)
    #
    # print(type(train_images[0]))
    # print(type(train_labels[0]))
    #
    # print(type(train_images[0][0]))
    # print(type(train_labels[0][0]))
    # exit()

    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(512, 512, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(4))


    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  # loss='CategoricalCrossentropy',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # print(history.history['accuracy'])
    # exit()



    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()

    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    y_predict = model.predict(test_images)

    print(test_labels.shape)
    print(y_predict.shape)

    for i in range(0, 34):
        print(y_predict[i], test_labels[i])




    # print("test_acc", test_acc)
    # print("test_loss", test_loss)


    # (18000, 150, 150)
    # (18000, 2)

    #
    # epoch = 20
    # # train_size_list = [0.2, 0.4]
    # train_size_list = [0.2, 0.4, 0.6, 0.8]
    #
    # X_data = X_data.reshape([-1, img_width, image_height, 1])/255
    #
    #
    # # set up cnn model
    # inputs = Input(shape=(150,150,1))
    # shared_layer = Conv2D(32, (3, 3), strides=2, activation='relu')(inputs)
    # shared_layer = MaxPooling2D((2, 2))(shared_layer)
    # shared_layer = Conv2D(64, (3, 3), strides=1, activation='relu')(shared_layer)
    # shared_layer = MaxPooling2D((2, 2))(shared_layer)
    # shared_layer = Conv2D(64, (3, 3), strides=1, activation='relu')(shared_layer)
    # shared_layer = Flatten()(shared_layer)
    #
    #
    # hour_layer = Dense(64, activation='relu')(shared_layer)
    # hour_layer = Dense(4, activation='softmax', name='specimen_type')(hour_layer)
    #
    # cls_model = Model(inputs=inputs, outputs=[hour_layer])
    # cls_model.compile(loss=['CategoricalCrossentropy'], optimizer='rmsprop', metrics=['accuracy'], run_eagerly=True)
    #
    #
    # # execute data set
    # classification_log_list = []
    #
    # for train_size in train_size_list:
    #     X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_label, train_size=train_size)
    #
    #
    #     history = cls_model.fit(X_train, Y_train, epochs=epoch)
    #
    #     Y_train_softmax_pred = cls_model.predict(X_test)
    #
    #     Y_hour_train_softmax_pred = np.array(Y_train_softmax_pred[0])
    #     Y_minutes_train_softmax_pred = np.array(Y_train_softmax_pred[1])
    #     Y_train_hr_pred = np.argmax(Y_hour_train_softmax_pred, axis=1)
    #     Y_train_minute_pred = np.argmax(Y_minutes_train_softmax_pred, axis=1)
    #     Y_train_hr_minute_pred = [Y_train_hr_pred[i]*60 + Y_train_minute_pred[i] for i in range(len(Y_train_hr_pred)) ]
    #
    #     Y_train_hr_accuracy = round(accuracy_score(Y_train_hour_label, Y_train_hr_pred), 4)
    #     Y_train_minute_accuracy = round(accuracy_score(Y_train_minute_label, Y_train_minute_pred), 4)
    #     Y_train_hr_minute_accuracy = round(accuracy_score(Y_train_hr_minute_label, Y_train_hr_minute_pred), 4)
    #     train_minute_loss = derive_minute_loss_from_classification(Y_train_hr_minute_label, Y_train_hr_minute_pred)
    #
    #
    #     Y_test_softmax_pred = cls_model.predict(X_test)
    #
    #     Y_hour_test_softmax_pred = np.array(Y_test_softmax_pred[0])
    #     Y_minutes_test_softmax_pred = np.array(Y_test_softmax_pred[1])
    #     Y_test_hr_pred = np.argmax(Y_hour_test_softmax_pred, axis=1)
    #     Y_test_minute_pred = np.argmax(Y_minutes_test_softmax_pred, axis=1)
    #     Y_test_hr_minute_pred = [Y_test_hr_pred[i]*60 + Y_test_minute_pred[i] for i in range(len(Y_test_hr_pred)) ]
    #
    #     Y_test_hr_accuracy = round(accuracy_score(Y_test_hour_label, Y_test_hr_pred), 4)
    #     Y_test_minute_accuracy = round(accuracy_score(Y_test_minute_label, Y_test_minute_pred), 4)
    #     Y_test_hr_minute_accuracy = round(accuracy_score(Y_test_hr_minute_label, Y_test_hr_minute_pred), 4)
    #     test_minute_loss = derive_minute_loss_from_classification(Y_test_hr_minute_label, Y_test_hr_minute_pred)
    #
    #
    #     log_msg = f"train_size: {train_size*100}%; Y_train_hr_minute_accuracy: {Y_train_hr_minute_accuracy}; Y_train_hr_accuracy: {Y_train_hr_accuracy}; Y_train_minute_accuracy: {Y_train_minute_accuracy}; train_minute_loss: {train_minute_loss}; Y_test_hr_minute_accuracy: {Y_test_hr_minute_accuracy}; Y_test_hr_accuracy: {Y_test_hr_accuracy}; Y_test_minute_accuracy: {Y_test_minute_accuracy}; test_minute_loss: {test_minute_loss}; "
    #     classification_log_list.append(log_msg)
    #     print(log_msg)
    #
    #     test_hour_cf = confusion_matrix(Y_test_hour_label, Y_test_hr_pred)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=test_hour_cf)
    #     disp.plot()


from matplotlib import pyplot

def obtain_cell_track_data_set(image_parent_dir: str):
    X_data_list = []
    Y_label_list = []

    sub_dir_list = os.listdir(image_parent_dir)

    from matplotlib import image
    for sub_dir in sub_dir_list:
        abs_sub_dir = image_parent_dir + sub_dir + "/"
        image_list: str = os.listdir(abs_sub_dir)
        for image_name in image_list:
            abs_image_file = abs_sub_dir + image_name
            image_data = image.imread(abs_image_file)
            # print(abs_sub_dir + image)

            # print(image_data.dtype)
            # print(image_data.shape)
            # pyplot.imshow(image_data)
            # pyplot.show()
            # print(type(image_data))

            X_data_list.append(image_data)

            # derive speciment type
            if sub_dir == "long_plus":  specimen_type = uint8(0)
            elif sub_dir == "long_minus":  specimen_type = uint8(1)
            elif sub_dir == "local_plus":  specimen_type = uint8(2)
            elif sub_dir == "local_minus":  specimen_type = uint8(3)
            else: raise Exception()


            Y_label_list.append([specimen_type])

    return np.asarray(X_data_list), np.asarray(Y_label_list)


def obtain_demo_data_set():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # print("train_images:", train_images.shape)
    # # print("train_labels:", train_labels[0:10])
    # exit()


    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # plt.figure(figsize=(10,10))
    # for i in range(1):
    #     plt.subplot(1,1,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i])
    #     # The CIFAR labels happen to be arrays,
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()

    return (train_images, train_labels), (test_images, test_labels)



def generate_np_array_from_batch_image(image_dir):
    image_file_list = os.listdir(image_dir)
    return ""

if __name__ == '__main__':
    main()