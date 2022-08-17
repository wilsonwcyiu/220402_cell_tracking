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

# train_accuracy:  0.8651
# test_accuracy:  0.4237
# [[180 103  45  26]
#  [ 94  79  79  21]
# [ 49  41 158  35]
# [ 29  33  72  44]]
#

def main():
    # config = tf.ConfigProto()   #https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size
    # config.gpu_options.allow_growth = True

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    img_parent_dir: str = "D:/viterbi linkage/dataset/track_classification_images_extended_rescale0.5/"
    # img_parent_dir: str = "D:/viterbi linkage/dataset/track_classification_images_extended/"
    # img_parent_dir: str = "D:/viterbi linkage/dataset/track_classification_images/"


    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()




    # (train_images, train_labels), (test_images, test_labels) = obtain_demo_data_set()
    train_set_ratio = 0.7
    image_length = 256


    X_data, Y_label = obtain_cell_track_data_set(img_parent_dir)
    total_img = X_data.shape[0]

    X_data = X_data.reshape((total_img, image_length, image_length, 1))
    X_data = X_data.astype('bool')
    Y_label = Y_label.reshape((Y_label.shape[0]))



    np.random.seed(16)
    shuffle_order = np.random.permutation(len(X_data))
    X_data = X_data[shuffle_order]
    Y_label = Y_label[shuffle_order]



    train_end = int(total_img * train_set_ratio)

    train_images = X_data[0: train_end]
    train_labels = Y_label[0: train_end]
    test_images = X_data[train_end: total_img]
    test_labels = Y_label[train_end: total_img]


    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(image_length, image_length, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))


    model.compile(optimizer='adam',
                  # loss='CategoricalCrossentropy',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])


    # history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
    history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))


    y_train_predict_arr_arr = model.predict(train_images)

    y_train_predict_list = []
    for y_train_predict_arr in y_train_predict_arr_arr:
        max_value = max(y_train_predict_arr)
        best_idx = y_train_predict_arr.tolist().index(max_value)
        y_train_predict_list.append(best_idx)

    train_accuracy = accuracy_score(train_labels, y_train_predict_list)
    train_accuracy = np.round(train_accuracy, 4)
    print("train_accuracy: ", train_accuracy)



    y_test_predict_arr_arr = model.predict(test_images)

    y_test_predict_list = []
    for y_test_predict_arr in y_test_predict_arr_arr:
        max_value = max(y_test_predict_arr)
        best_idx = y_test_predict_arr.tolist().index(max_value)
        y_test_predict_list.append(best_idx)

    # calculate accuracy
    test_accuracy = accuracy_score(test_labels, y_test_predict_list)
    test_accuracy = np.round(test_accuracy, 4)
    print("test_accuracy: ", test_accuracy)



    from sklearn.metrics import confusion_matrix
    cf_matrix = confusion_matrix(test_labels, y_test_predict_list)

    print(cf_matrix)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    #https://www.stackvidhya.com/plot-confusion-matrix-in-python-and-why/#:~:text=Plot%20Confusion%20Matrix%20for%20Binary%20Classes%20With%20Labels&text=You%20need%20to%20create%20a,matrix%20with%20the%20labels%20annotation.

    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Zebra fish cell image classification\n');
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Distant +/+', 'Distant -/-', 'Local +/+', 'Local -/-'])
    ax.yaxis.set_ticklabels(['Distant +/+', 'Distant -/-', 'Local +/+', 'Local -/-'])

    ## Display the visualization of the Confusion Matrix.
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=['Long\nPlus','Long\nMinus', 'Local\nPlus', 'Local\nMinus'])
    disp.plot(cmap="Blues", values_format='d')
    plt.show()



def obtain_cell_track_data_set(image_parent_dir: str):
    X_data_list = []
    Y_label_list = []

    sub_dir_list = os.listdir(image_parent_dir)

    img_cnt = 0
    from matplotlib import image
    for sub_dir in sub_dir_list:
        abs_sub_dir = image_parent_dir + sub_dir + "/"
        image_list: str = os.listdir(abs_sub_dir)
        for image_name in image_list:
            img_cnt += 1
            print("\rimg count", img_cnt, end='')
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