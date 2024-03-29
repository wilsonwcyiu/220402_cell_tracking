# -*- coding: utf-8 -*-
"""Lineage_methods_Viterbi_visualize.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZAMApLe-8OPQt3EC0MwOK5r3EcYe_a-d
"""

# from google.colab import drive

import os
from collections import namedtuple, defaultdict
from datetime import datetime
from multiprocessing.pool import ThreadPool
from os import listdir
from os.path import join, basename
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, filters
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
import copy
import time
import cv2
# from google.colab.patches import cv2_imshow
import random
from itertools import combinations
import pickle

from other.shared_cell_data import obtain_ground_truth_cell_dict
import csv
import pandas as pd

ROOT = '/content/drive/'     # default for the drive
# drive.mount(ROOT)           # we mount the drive at /content/drive

# PROJECT_PATH = '/content/drive/MyDrive'

"""# 新段落"""

def main():
    folder_path: str = 'D:/viterbi linkage/dataset/'

    rescale = 0.5

    save_dir = folder_path + 'track_classification_images_extended_rescale' + str(rescale) + '/'
    classification_csv_folder_path = folder_path + "track_classification/"
    # local_category_csv_folder_path = folder_path + "track_classification/"

    pkl_file_name: str = "viterbi_adjust4f_hp001.pkl"
    output_dir_name: str = pkl_file_name.replace(".pkl", "")

    #settings
    # fixed_track_length_to_generate = 16
    point_radius_pixel = 1



    rotation_list = [0, 90, 180, 270]

    x_shift_list = [-50, 0, 50]
    y_shift_list = [-50, 0, 50]

    x_shift_list = (np.asarray(x_shift_list) * rescale).astype('int')
    y_shift_list = (np.asarray(y_shift_list) * rescale).astype('int')


    img_length = int(512 * rescale)
    img_dimension_tuple: tuple = (img_length, img_length)

    # x_shift_list = [0]
    # y_shift_list = [0]

    xy_shift_tuple_list = []
    for x_shift in x_shift_list:
        for y_shift in y_shift_list:
            xy_shift_tuple_list.append((x_shift, y_shift))

    # print(xy_shift_tuple_list)
    # print(len(xy_shift_tuple_list))
    # exit()

    start_time = time.perf_counter()
    date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S")


    for movement_type in ["long", "local"]:
        csv_file_list = os.listdir(classification_csv_folder_path + movement_type)
        for csv_file_name in csv_file_list:
            print("handling file: ", csv_file_name)

            polar_type = "minus" if "--" in csv_file_name else "plus"

            df = pd.read_csv(classification_csv_folder_path + movement_type + "/"+ csv_file_name)

            # create data
            track_frame_num_coord_dict_dict = defaultdict(dict)
            for index, row in df.iterrows():
                track_num = int(row[0])
                frame_num = int(row[1])
                x, y = row[2], row[3]
                new_x = int(np.round(x * rescale, 0))
                new_y = int(np.round(y * rescale, 0))
                track_frame_num_coord_dict_dict[track_num][frame_num] = CoordTuple(new_x, new_y)



            # generate image from data
            for track_num, frame_num_coord_dict in track_frame_num_coord_dict_dict.items():

                for xy_shift_tuple in xy_shift_tuple_list:
                    is_valid_shift, new_frame_num_coord_dict = shift_coord(frame_num_coord_dict, xy_shift_tuple)

                    if not is_valid_shift:
                        continue

                    coord_tuple_list = []
                    for frame_num, coord in new_frame_num_coord_dict.items():
                        coord_tuple_list.append(coord)

                    result_folder = save_dir + movement_type + "_" + polar_type + "/"
                    if not os.path.exists(result_folder):   os.makedirs(result_folder)

                    file_name = csv_file_name.replace(".csv", "").replace(" ", "_") + "_" + str(track_num)


                    for rotation in rotation_list:
                        abs_file_path = result_folder + file_name + f"_x{xy_shift_tuple[0]}_y{xy_shift_tuple[1]}_r{rotation}.png"

                        # print(abs_file_path)
                        generate_track_image(coord_tuple_list, point_radius_pixel, abs_file_path, img_dimension_tuple, line_width = 1, rotate=rotation)



def shift_coord(frame_num_coord_dict, xy_shift_tuple):
    is_valid_shift = True
    new_frame_num_coord_dict = {}

    for frame_num, coord in frame_num_coord_dict.items():
        new_x = coord.x + xy_shift_tuple[0]
        new_y = coord.y + xy_shift_tuple[1]

        if new_x < 0 or new_x > 512 or new_y < 0 or new_y > 512:
            is_valid_shift = False
            # print("invalid shift", xy_shift_tuple, new_x, new_y)
            break

        else:
            new_frame_num_coord_dict[frame_num] = CoordTuple(new_x, new_y)

    return is_valid_shift, new_frame_num_coord_dict



CoordTuple = namedtuple("CoordTuple", "x y")

def generate_track_image(coord_tuple_list: list, point_radius_pixel: float, abs_file_path: str, img_dimension_tuple: tuple, line_width = 1, rotate=None):

    background = 1
    line_color = 0

    im = Image.new('1', img_dimension_tuple, background)
    draw = ImageDraw.Draw(im)

    total_point: int = len(coord_tuple_list)
    second_last_point = total_point - 1
    for track_idx in range(0, second_last_point):
        current_coord = coord_tuple_list[track_idx]
        next_coord = coord_tuple_list[track_idx+1]

        draw.line((current_coord.x, current_coord.y, next_coord.x, next_coord.y), fill=line_color, width=line_width)

        left_top_coord = (current_coord.x - point_radius_pixel, current_coord.y - point_radius_pixel)
        right_bottom_coord = (current_coord.x + point_radius_pixel, current_coord.y + point_radius_pixel)
        draw.ellipse([left_top_coord, right_bottom_coord], fill=line_color)

        left_top_coord = (next_coord.x - point_radius_pixel, next_coord.y - point_radius_pixel)
        right_bottom_coord = (next_coord.x + point_radius_pixel, next_coord.y + point_radius_pixel)
        draw.ellipse([left_top_coord, right_bottom_coord], fill=line_color)

    if rotate is not None:
        im = im.rotate(rotate)

    im.save(abs_file_path)







if __name__ == '__main__':
    main()