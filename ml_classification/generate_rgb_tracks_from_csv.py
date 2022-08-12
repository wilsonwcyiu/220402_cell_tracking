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

    save_dir = folder_path + 'track_classification_images/'
    classification_csv_folder_path = folder_path + "track_classification/"
    # local_category_csv_folder_path = folder_path + "track_classification/"

    pkl_file_name: str = "viterbi_adjust4f_hp001.pkl"
    output_dir_name: str = pkl_file_name.replace(".pkl", "")



    #settings
    # fixed_track_length_to_generate = 16
    point_radius_pixel = 1



    start_time = time.perf_counter()
    date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # long_csv_file_list = os.listdir(long_category_csv_folder_path)
    # local_csv_file_list = os.listdir(local_category_csv_folder_path)

    # print("dsfg", long_csv_file_list[0])


    # abs_dir_path = f"{save_dir}{date_str}_cnn_track_data_from_csv/"
    # if not os.path.exists(abs_dir_path):
    #     print(f"create abs_dir_path: {abs_dir_path}")
    #     os.makedirs(abs_dir_path)




    plus_long_track_coord_list = []
    plus_local_track_coord_list = []
    minus_long_track_coord_list = []
    minus_local_track_coord_list = []


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
                track_frame_num_coord_dict_dict[track_num][frame_num] = CoordTuple(x, y)

            # generate image from data
            for track_num, frame_num_coord_dict in track_frame_num_coord_dict_dict.items():
                coord_tuple_list = []
                for frame_num, coord in frame_num_coord_dict.items():
                    coord_tuple_list.append(coord)

                result_folder = save_dir + movement_type + "_" + polar_type + "/"
                file_name = csv_file_name.replace(".csv", "").replace(" ", "_") + "_" + str(track_num)
                abs_file_path = result_folder + file_name + ".png"

                # print(abs_file_path)
                generate_track_image(coord_tuple_list, point_radius_pixel, abs_file_path, line_width = 1)




CoordTuple = namedtuple("CoordTuple", "x y")

def generate_track_image(coord_tuple_list: list, point_radius_pixel: float, abs_file_path: str, line_width = 1):
    img_dimension: tuple = (512, 512)
    background_rgb = (255, 255, 255)
    line_rgb = (0, 0, 0)

    im = Image.new('RGB', img_dimension, background_rgb)
    draw = ImageDraw.Draw(im)

    total_point: int = len(coord_tuple_list)
    second_last_point = total_point - 1
    for track_idx in range(0, second_last_point):
        current_coord = coord_tuple_list[track_idx]
        next_coord = coord_tuple_list[track_idx+1]

        draw.line((current_coord.x, current_coord.y, next_coord.x, next_coord.y), fill=line_rgb, width=line_width)

        left_top_coord = (current_coord.x - point_radius_pixel, current_coord.y - point_radius_pixel)
        right_bottom_coord = (current_coord.x + point_radius_pixel, current_coord.y + point_radius_pixel)
        draw.ellipse([left_top_coord, right_bottom_coord], fill=line_rgb)

        left_top_coord = (next_coord.x - point_radius_pixel, next_coord.y - point_radius_pixel)
        right_bottom_coord = (next_coord.x + point_radius_pixel, next_coord.y + point_radius_pixel)
        draw.ellipse([left_top_coord, right_bottom_coord], fill=line_rgb)

    im.save(abs_file_path)







if __name__ == '__main__':
    main()