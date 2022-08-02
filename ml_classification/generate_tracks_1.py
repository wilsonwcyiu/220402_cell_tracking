# -*- coding: utf-8 -*-
"""Lineage_methods_Viterbi_visualize.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZAMApLe-8OPQt3EC0MwOK5r3EcYe_a-d
"""

# from google.colab import drive

import os
from collections import namedtuple
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

ROOT = '/content/drive/'     # default for the drive
# drive.mount(ROOT)           # we mount the drive at /content/drive

# PROJECT_PATH = '/content/drive/MyDrive'

"""# 新段落"""

def main():
    folder_path: str = 'D:/viterbi linkage/dataset/'

    segmentation_folder = folder_path + 'segmentation_unet_seg//'
    save_dir = folder_path + '/save_directory_enhancement/'
    video_folder_name = folder_path + '/save_directory_enhancement/trajectory_result_video/'
    pkl_file_name: str = "viterbi_adjust4f_hp001.pkl"
    output_dir_name: str = pkl_file_name.replace(".pkl", "")



    #settings
    filter_out_track_length_lower_than: int = 16
    fixed_track_length_to_generate = 16
    point_radius_pixel = 1



    to_generate_series_list: list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                                     'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']

    start_time = time.perf_counter()
    date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # for series in to_generate_series_list:
    all_segmented_filename_list = listdir(segmentation_folder)
    all_segmented_filename_list.sort()


    series_viterbi_result_list_dict: dict = open_track_dictionary(save_dir + pkl_file_name)

    abs_dir_path = f"{save_dir}{date_str}_cnn_track_data/"
    if not os.path.exists(abs_dir_path):
        print(f"create abs_dir_path: {abs_dir_path}")
        os.makedirs(abs_dir_path)


    for series, result_list_list in series_viterbi_result_list_dict.items():
        print("working on series:", series)

        result_list_list = sorted(result_list_list)

        segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)

        frame_num_node_id_coord_dict_dict = derive_frame_num_node_idx_coord_list_dict(segmentation_folder, segmented_filename_list)

        # filter result_list that is at least given length
        result_list_list = filter_track_by_length(result_list_list, filter_out_track_length_lower_than)

        result_list_list = generate_all_combination_fixed_track_length(result_list_list, fixed_track_length_to_generate)

        for idx, result_list in enumerate(result_list_list):
            print(f"\r{idx+1}/ {len(result_list_list)}; ", end='')
            coord_tuple_list = []
            for result_tuple in result_list:
                cell_idx = result_tuple[0]
                frame_num = result_tuple[1] + 1
                coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[frame_num][cell_idx]
                coord_tuple_list.append(coord_tuple)

            abs_file_path = f"{abs_dir_path}{series}_{idx}.png"
            generate_track_image(coord_tuple_list, point_radius_pixel, abs_file_path)

        print()

    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {execution_time: 0.4f} seconds")



def derive_segmented_filename_list_by_series(series: str, segmented_filename_list: list):
    result_segmented_filename_list: list = []

    for segmented_filename in segmented_filename_list:
        if series in segmented_filename:
            result_segmented_filename_list.append(segmented_filename)

    return result_segmented_filename_list


CoordTuple = namedtuple("CoordTuple", "x y")

def derive_frame_num_node_idx_coord_list_dict(segmentation_folder_path: str, segmented_filename_list):
    frame_num_cell_coord_list_dict: dict = {}

    for frame_idx in range(0, len(segmented_filename_list)):
        frame_num = frame_idx + 1
        img = plt.imread(segmentation_folder_path + segmented_filename_list[frame_idx])
        label_img = measure.label(img, background=0, connectivity=1)
        cellnb_img = np.max(label_img)

        cell_coord_tuple_list = []
        for cellnb_i in range(cellnb_img):

            cell_i = plt.imread(segmentation_folder_path + segmented_filename_list[0])
            cell_i_props = measure.regionprops(label_img, intensity_image=cell_i) #label_img_next是二值图像为255，无intensity。需要与output中的预测的细胞一一对应，预测细胞有intensity

            y, x = cell_i_props[cellnb_i].centroid
            y, x = int(y), int(x)

            cell_coord_tuple_list.append(CoordTuple(x, y))

        frame_num_cell_coord_list_dict[frame_num] = cell_coord_tuple_list

    return frame_num_cell_coord_list_dict



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



'''opening a track dictionary from file'''
def open_track_dictionary(save_file):
    pickle_in = open(save_file, "rb")
    dictionary = pickle.load(pickle_in)

    return dictionary


"""Draw and plot tracks"""

'''draw and save plotted tracks'''
np.set_printoptions(edgeitems=30, linewidth=100000)

#get a slightly darker colour for the points indicating the current track
def get_point_color(color):
    point_color = tuple(np.subtract(color, (20,20,20)))
    point_color= tuple([int(i) for i in point_color])
    return point_color



#print (& save) the cell tracks in each frame for a max number of TRACK_LENGTH frames
def draw_and_save_tracks(to_generate_series_list, series_viterbi_result_list_dict, segmentation_folder, video_folder, is_save_result: bool, to_print_track_length: int, dir_name: str, filter_out_track_length_lower_than: int, fixed_track_length_to_generate: int):
    generation_mode: str = "single"  #{single, all}
    is_use_thread: bool = False
    if is_use_thread:

        pool = ThreadPool(processes=8)
        thread_list: list = []

        for series, result_list_list in series_viterbi_result_list_dict.items():
            if series not in to_generate_series_list:
                continue

            print("working on series:", series)


            args_tuple: tuple = (series, result_list_list, segmentation_folder, video_folder, is_save_result, to_print_track_length, dir_name,) # tuple of args for foo
            async_result = pool.apply_async(draw_and_save_tracks_single, args_tuple)
            thread_list.append(async_result)

        total_threads = len(thread_list)
        for thread_idx in range(total_threads):
            thread_list[thread_idx].get()
            print(f"Thread {thread_idx+1}/{total_threads}, series {series} completed")

    else:
        ground_truth_cell_dict = obtain_ground_truth_cell_dict()
        if generation_mode == "all":
            for series, result_list_list in series_viterbi_result_list_dict.items():
                print("working on series:", series)


                # is_only_show_ground_truth_related_track = False
                # if is_only_show_ground_truth_related_track:
                #     ground_truth_cell_list = ground_truth_cell_dict[series]
                #     filtered_list_list = []
                #     for result_list in result_list_list:
                #         if result_list[0] in ground_truth_cell_list:
                #             filtered_list_list.append(result_list)
                #     result_list_list = filtered_list_list

                draw_and_save_tracks_single(series, result_list_list, segmentation_folder, video_folder, is_save_result, to_print_track_length, dir_name)

        elif generation_mode == "single":
            for series, result_list_list in series_viterbi_result_list_dict.items():
                print("working on series:", series)

                result_list_list = sorted(result_list_list)

                # filter result_list that is at least given length
                result_list_list = filter_track_by_length(result_list_list, filter_out_track_length_lower_than)

                result_list_list = generate_all_combination_fixed_track_length(result_list_list, fixed_track_length_to_generate)


                for idx, result_list in enumerate(result_list_list):
                    print(f"{idx}/ {len(result_list_list)}; ")
                    cell_idx = result_list[0][0]
                    print("cell_idx", cell_idx, end='')
                    dir_suffix = "_cell_idx_" + str(cell_idx)
                    draw_and_save_tracks_single(series, [result_list], segmentation_folder, video_folder, is_save_result, to_print_track_length, dir_name, dir_suffix=dir_suffix)
        else:
            raise Exception()



def generate_all_combination_fixed_track_length(result_list_list, fixed_track_length_to_generate):
    all_fixed_track_length_tuple_list_list = []

    for result_list in result_list_list:
        track_length = len(result_list)
        num_of_track_to_generate = track_length - fixed_track_length_to_generate + 1
        for start_idx in range(0, num_of_track_to_generate):
            fixed_track_length_tuple_list = []
            end_track = start_idx + fixed_track_length_to_generate
            for idx in range(start_idx, end_track):
                fixed_track_length_tuple_list.append(result_list[idx])

            all_fixed_track_length_tuple_list_list.append(fixed_track_length_tuple_list)

    return all_fixed_track_length_tuple_list_list






def filter_track_by_length(result_list_list, filter_out_track_length_lower_than):
    filtered_result_list_list = []
    for track_tuple_list in result_list_list:
        if len(track_tuple_list) < filter_out_track_length_lower_than:
            continue

        filtered_result_list_list.append(track_tuple_list)
        
    return filtered_result_list_list



def draw_and_save_tracks_single(series,
                                result_list,
                                segmentation_folder,
                                video_folder,
                                is_save_result: bool,
                                track_length_to_print,
                                dir_name: str,
                                dir_suffix: str=""):

    #get the correct segementation files
    segmented_files = listdir(segmentation_folder)
    segmented_files.sort()
    seg_file_list = []

    #select all files of the current images series and celltype
    for filename in segmented_files:
        if series in filename:
            seg_file_list = seg_file_list + [filename]

    img_list = []
    labeled_img_list = []
    for frame_idx in range(len(seg_file_list)):
        img = cv2.imread(segmentation_folder + '/' + seg_file_list[frame_idx])
        label_img = measure.label(img, background=0,connectivity=1)
        img_list.append(img)
        labeled_img_list.append(label_img)

    #create a list to select random colours for tracks from
    colorlist = []
    colorlist.append((255, 255, 255))
    colorlist.append((255, 0, 0))
    colorlist.append((255, 165, 0))
    colorlist.append((255, 255, 0))
    colorlist.append((0, 128, 0))
    colorlist.append((0, 0, 255))
    colorlist.append((75, 0, 130))
    colorlist.append((238, 130, 238))
    random.seed(23)
    for i in range(len(result_list)):
        r = random.randint(30,255)
        g = random.randint(30,255)
        b = random.randint(30,255)
        colorlist.append((r,g,b))

    #loop through all tracked frames to create the tracked images
    for frame_idx in range(len(seg_file_list)):
        # frame_idx = frame_idx
        # if frame_idx < 110:
        #     print(f"skip frame {frame_idx};", end='')
        #     continue


        print(f"{frame_idx},", end='')

        img = img_list[frame_idx]
        for track_idx, track_tuple_list in enumerate(result_list):

            #debug
            if track_idx != 19:
                continue

            print("sdfvsdf", len(track_tuple_list))
            print("dfgf", track_tuple_list)



            frames_in_track = np.array(list(zip(*track_tuple_list))[1])

            #do not print track_tuple_list if not in current frame
            if not frame_idx in frames_in_track:
                continue

            #get part of current track_tuple_list for a maximum of TRACK_LENGTH frames back to the current frame
            partial_track_idx = np.where((frames_in_track <= frame_idx) & (frames_in_track >= frame_idx - track_length_to_print))

            #print line piece for every cell in partial track_tuple_list
            for cell_idx in partial_track_idx[0]:
                cell_idx = int(cell_idx)
                props = measure.regionprops(labeled_img_list[track_tuple_list[cell_idx][1]])
                cell_coord_y, cell_coord_x,_ = props[track_tuple_list[cell_idx][0]].centroid

                #print a dot at current cell position
                if track_tuple_list[cell_idx][1] == frame_idx:
                    cv2.circle(img, (int(cell_coord_x), int(cell_coord_y)), 2, get_point_color(colorlist[track_idx]), -1)
                    # cv2.putText(img, str(track_tuple_list[cell_idx][0]), (int(cell_coord_x),int(cell_coord_y-10)), cv2.FONT_HERSHEY_COMPLEX,0.4,(147,20,255),1)

                #skip first frame
                if track_tuple_list[cell_idx][2] == -1:
                    continue

                #get coordinates of previous cellposition from track_tuple_list
                i = 1
                while True:
                    print("\rvv", cell_idx, end='')
                    if track_tuple_list[cell_idx][2] == track_tuple_list[cell_idx-i][0]:
                        props = measure.regionprops(labeled_img_list[track_tuple_list[cell_idx][1]-1])
                        prev_cell_coord_y, prev_cell_coord_x,_ = props[track_tuple_list[cell_idx-i][0]].centroid
                        break
                    else:
                        i = i+1

                cv2.line(img, (int(prev_cell_coord_x), int(prev_cell_coord_y)), (int(cell_coord_x), int(cell_coord_y)), colorlist[track_idx])

        # cv2_imshow(img)
        #save the created images
        if is_save_result:
            # output_dir_path: str = video_folder + dir_name + "/" + str(series) + "/"
            output_dir_path: str = video_folder + dir_name + "/" + str(series) + dir_suffix + "/"
            if not os.path.isdir(output_dir_path):
                os.makedirs(output_dir_path)
            cv2.imwrite(output_dir_path + '/' + str(frame_idx) +'.png', img)
    print(" ---> end")







if __name__ == '__main__':
    main()