
import os

import cv2

import os

from PIL import Image


def main():
    dataset_dir = "D://viterbi linkage//dataset//"

    img_parent_dir: str = dataset_dir + "track_classification_images_extended//"
    output_dir = dataset_dir + "track_classification_images_rescaled//"

    downscale_image(img_parent_dir, output_dir)
    # img = cv2.imread('data.png', 1)
    # cv2.imshow('Original', img)
    #
    # img_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow('Half Image', img_half)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def downscale_image(image_parent_dir: str, output_parent_dir: str):
    X_data_list = []
    Y_label_list = []
    size = 128,128

    sub_dir_list = os.listdir(image_parent_dir)

    img_cnt = 0
    from matplotlib import image
    for sub_dir in sub_dir_list:
        abs_input_sub_dir = image_parent_dir + sub_dir + "/"

        abs_output_sub_dir = output_parent_dir + sub_dir + "/"
        if not os.path.exists(abs_output_sub_dir):
            os.makedirs(abs_output_sub_dir)

        image_list: str = os.listdir(abs_input_sub_dir)
        for image_name in image_list:
            img_cnt += 1
            print("img count", img_cnt)
            abs_input_image_file = abs_input_sub_dir + image_name

            if not os.path.exists(abs_input_image_file):
                raise Exception()

            abs_output_image_file = abs_output_sub_dir + image_name

            im = Image.open(abs_input_image_file)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(abs_output_image_file, "png")

            # img = cv2.imread(abs_input_image_file)
            # # cv2.imshow('Original', img)
            # img_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            #
            # abs_output_image_file = abs_output_sub_dir + image_name
            # cv2.imwrite(abs_output_image_file, img_half)





if __name__ == '__main__':
    main()