# -*- coding: utf-8 -*-
"""Lineage_methods_Viterbi_visualize.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZAMApLe-8OPQt3EC0MwOK5r3EcYe_a-d
"""

# from google.colab import drive
import os
from multiprocessing.pool import ThreadPool
from os import listdir
from os.path import join, basename
import numpy as np
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
    pkl_file_name: str = "ground_truth_results_dict.pkl"




    #settings
    track_length: int = 120
    is_save_result = True

    to_generate_series_list: list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                                     'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
    to_generate_series_list = ["S11"]
    # to_generate_series_list1: list = ['S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19']
    # fail_to_generate_series_list = ['S04', 'S07', 'S08', 'S09', 'S12', 'S20' ]
    # to_generate_series_list: list = ['S01', 'S02', 'S03', 'S05', 'S06', 'S07', 'S08', 'S10',
    #                                  'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']

    # fail_to_generate_series_list = ['S04']

    start_time = time.perf_counter()

    series_viterbi_result_list_dict: dict = open_track_dictionary(save_dir + pkl_file_name)


    # for series, track_list_list in series_viterbi_result_list_dict.items():
    #     length_info_list = []
    #     for track_list in track_list_list:
    #         length_info_list.append(len(track_list))
    #
    #     length_info_list.sort()
    #     print(series, length_info_list)
    # exit()


    output_dir_name: str = pkl_file_name.replace(".pkl", "")
    draw_and_save_tracks(to_generate_series_list, series_viterbi_result_list_dict, segmentation_folder, video_folder_name, is_save_result, track_length, output_dir_name)


    execution_time = time.perf_counter() - start_time
    print(f"Execution time: {execution_time: 0.4f} seconds")



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
def draw_and_save_tracks(to_generate_series_list, series_viterbi_result_list_dict, segmentation_folder, video_folder, is_save_result: bool, track_length: int, dir_name: str):
    generation_mode: str = "single"  #{single, all}
    is_use_thread: bool = False
    if is_use_thread:

        pool = ThreadPool(processes=8)
        thread_list: list = []

        for series, result_list_list in series_viterbi_result_list_dict.items():
            if series not in to_generate_series_list:
                continue

            print("working on series:", series)


            args_tuple: tuple = (series, result_list_list, segmentation_folder, video_folder, is_save_result, track_length, dir_name, ) # tuple of args for foo
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
                if series not in to_generate_series_list:
                    continue
                print("working on series:", series)


                is_only_show_ground_truth_related_track = False
                if is_only_show_ground_truth_related_track:
                    ground_truth_cell_list = ground_truth_cell_dict[series]
                    filtered_list_list = []
                    for result_list in result_list_list:
                        if result_list[0] in ground_truth_cell_list:
                            filtered_list_list.append(result_list)
                    result_list_list = filtered_list_list

                draw_and_save_tracks_single(series, result_list_list, segmentation_folder, video_folder, is_save_result, track_length, dir_name)

        elif generation_mode == "single":
            for series, result_list_list in series_viterbi_result_list_dict.items():
                if series not in to_generate_series_list:
                    continue
                print("working on series:", series)

                result_list_list = sorted(result_list_list)
                for result_list in result_list_list:
                    cell_idx = result_list[0][0]



                    print("cell_idx", cell_idx, ". ", end='')
                    dir_suffix = "_cell_idx_" + str(cell_idx)
                    draw_and_save_tracks_single(series, [result_list], segmentation_folder, video_folder, is_save_result, track_length, dir_name, dir_suffix=dir_suffix)
        else:
            raise Exception()






def draw_and_save_tracks_single(series, result_list, segmentation_folder, video_folder, is_save_result: bool, TRACK_LENGTH, dir_name: str, dir_suffix: str=""):
    #get the correct segementation files
    segmented_files = listdir(segmentation_folder)
    segmented_files.sort()
    file_list = []

    #select all files of the current images series and celltype
    for filename in segmented_files:
        if series in filename:
            file_list = file_list + [filename]

    img_list = []
    labeled_img_list = []
    for framenb in range(len(file_list)):
        img = cv2.imread(segmentation_folder + '/' + file_list[framenb])
        label_img = measure.label(img, background=0,connectivity=1)
        img_list.append(img)
        labeled_img_list.append(label_img)

    #create a list to select random colours for tracks from
    colorlist = []
    colorlist.append((0, 0, 255))
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
    for framenb in range(len(file_list)):
        frame_idx = framenb
        # if frame_idx < 110:
        #     print(f"skip frame {frame_idx};", end='')
        #     continue


        print(f"{framenb},", end='')

        img = img_list[framenb]
        for tracknb, track in enumerate(result_list):

            frames_in_track = np.array(list(zip(*track))[1])

            #do not print track if not in current frame
            if not framenb in frames_in_track:
                continue

            #get part of current track for a maximum of TRACK_LENGTH frames back to the current frame
            partial_track_idx = np.where((frames_in_track <= framenb) & (frames_in_track >= framenb-TRACK_LENGTH))

            #print line piece for every cell in partial track
            for cell_idx in partial_track_idx[0]:
                cell_idx = int(cell_idx)
                props = measure.regionprops(labeled_img_list[track[cell_idx][1]])
                cell_coord_y, cell_coord_x,_ = props[track[cell_idx][0]].centroid

                #print a dot at current cell position
                if track[cell_idx][1] == framenb:
                    cv2.circle(img, (int(cell_coord_x), int(cell_coord_y)), 2, get_point_color(colorlist[tracknb]), -1)
                    # cv2.putText(img, str(track[cell_idx][0]), (int(cell_coord_x),int(cell_coord_y-10)), cv2.FONT_HERSHEY_COMPLEX,0.4,(147,20,255),1)

                #skip first frame
                if track[cell_idx][2] == -1:
                    continue

                #get coordinates of previous cellposition from track
                i = 1
                while True:
                    if track[cell_idx][2] == track[cell_idx-i][0]:
                        props = measure.regionprops(labeled_img_list[track[cell_idx][1]-1])
                        prev_cell_coord_y, prev_cell_coord_x,_ = props[track[cell_idx-i][0]].centroid
                        break
                    else:
                        i = i+1

                cv2.line(img, (int(prev_cell_coord_x), int(prev_cell_coord_y)), (int(cell_coord_x), int(cell_coord_y)), colorlist[tracknb])
        # cv2_imshow(img)
        #save the created images
        if is_save_result:
            # output_dir_path: str = video_folder + dir_name + "/" + str(series) + "/"
            output_dir_path: str = video_folder + dir_name + "/" + str(series) + dir_suffix + "/"
            if not os.path.isdir(output_dir_path):
                os.makedirs(output_dir_path)
            cv2.imwrite(output_dir_path + '/' + str(framenb) +'.png', img)
    print(" ---> end")







if __name__ == '__main__':
    main()