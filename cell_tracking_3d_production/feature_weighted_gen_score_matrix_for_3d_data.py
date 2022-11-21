"""
First version: 16 May 2022
@author: Wilson W.C. Yiu
"""

import enum
import itertools
import math
import os
import pickle
import time
import traceback
from collections import defaultdict, namedtuple
from datetime import datetime
from functools import cmp_to_key
from math import atan2, degrees, radians, cos
from pathlib import Path
import numpy as np
from imutils import paths
import SimpleITK as sitk


def main():



    ## settings
    folder_path: str = 'D:/viterbi linkage/dataset/'
    raw_3d_data_dir: str = folder_path + '3D raw data_seg data_find center coordinate/4 Segmentation dataset/'
    default_save_dir = folder_path + 'save_directory_enhancement/'

    ## user input
    input_series_name_sub_dir_dict: dict = {}
    # input_series_name_sub_dir_dict["20190621++2_8layers_M3a_Step98"] = "1 8layers mask data/20190621++2_8layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20190701--1_8layers_M3a_Step98"] = "1 8layers mask data/20190701--1_8layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20190701--2_8layers_M3a_Step98"] = "1 8layers mask data/20190701--2_8layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20200716++1_8layers_M3s_Step98"] = "1 8layers mask data/20200716++1_8layers_M3s_Step98"
    # input_series_name_sub_dir_dict["20200716++2_8layers_M3a_Step98"] = "1 8layers mask data/20200716++2_8layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20200802--2_9layers_M3a_Step98"] = "2 9layers mask data/20200802--2_9layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20200829++1_9layers_M3a_Step98"] = "2 9layers mask data/20200829++1_9layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20200829++2_9layers_M3a_Step98"] = "2 9layers mask data/20200829++2_9layers_M3a_Step98"
    # input_series_name_sub_dir_dict["20200829--1_9layers_M3a_Step98"] = "2 9layers mask data/20200829--1_9layers_M3a_Step98"

    input_series_name_sub_dir_dict["20190621++2_inter_29layers_mask_3a"] = "5 29layers inter mask data/model3a/20190621++2_inter_29layers_mask_3a"
    input_series_name_sub_dir_dict["20190701--1_inter_29layers_mask_3a"] = "5 29layers inter mask data/model3a/20190701--1_inter_29layers_mask_3a"
    input_series_name_sub_dir_dict["20190701--2_inter_29layers_mask_3a"] = "5 29layers inter mask data/model3a/20190701--2_inter_29layers_mask_3a"
    input_series_name_sub_dir_dict["20200716++1_inter_29layers_mask_3a"] = "5 29layers inter mask data/model3a/20200716++1_inter_29layers_mask_3a"
    input_series_name_sub_dir_dict["20200716++2_inter_29layers_mask_3a"] = "5 29layers inter mask data/model3a/20200716++2_inter_29layers_mask_3a"
    input_series_name_sub_dir_dict["20200802--2_inter_33layers_mask_3a"] = "6 33layers inter mask data/20200802--2_inter_33layers_mask_3a"
    input_series_name_sub_dir_dict["20200829++1_inter_33layers_mask_3a"] = "6 33layers inter mask data/20200829++1_inter_33layers_mask_3a"
    input_series_name_sub_dir_dict["20200829++2_inter_33layers_mask_3a"] = "6 33layers inter mask data/20200829++2_inter_33layers_mask_3a"
    input_series_name_sub_dir_dict["20200829--1_inter_33layers_mask_3a"] = "6 33layers inter mask data/20200829--1_inter_33layers_mask_3a"

    ## hyper parameter settings
    weight_tuple_list: list = [WeightTuple(0.3, 0.4, 0.3)]
    max_moving_distance_list: list = [40]
    coord_length_for_vector_list: list = [6]
    average_movement_step_length_list: list = [6]
    minimum_track_length_list: list = [1]

    ## end of user input






    discount_rate_per_layer_list: list = [None]      # depricated

    start_datetime: Date = datetime.now()
    date_str: str = start_datetime.strftime("%Y%m%d-%H%M%S")
    py_file_name: str = Path(__file__).name.replace(".py", "")
    individual_result_dir: str = default_save_dir + date_str + "_" + py_file_name + "/"
    os.makedirs(individual_result_dir)

    score_log_dir: str = individual_result_dir + "score_log/"
    os.makedirs(score_log_dir)

    hyper_para_combination_list = list(itertools.product(
                                                         weight_tuple_list,
                                                         max_moving_distance_list,
                                                         coord_length_for_vector_list,
                                                         average_movement_step_length_list,
                                                         minimum_track_length_list,
                                                         discount_rate_per_layer_list
                                                         ))

    hyper_para_list: list = []
    for hyper_para_combination in hyper_para_combination_list:
        weight_tuple = hyper_para_combination[0]
        max_moving_distance = hyper_para_combination[1]
        coord_length_for_vector = hyper_para_combination[2]
        average_movement_step_length = hyper_para_combination[3]
        minimum_track_length: int = hyper_para_combination[4]
        discount_rate_per_layer: [str, float] = hyper_para_combination[5]

        hyper_para: HyperPara = HyperPara(
                                            weight_tuple,
                                            max_moving_distance,
                                            coord_length_for_vector,
                                            average_movement_step_length,
                                            minimum_track_length,
                                            discount_rate_per_layer)

        hyper_para_list.append(hyper_para)


    total_para_size: int = len(hyper_para_list)
    for hyperpara_idx, hyper_para in enumerate(hyper_para_list):
        para_set_num: int = hyperpara_idx + 1

        print(f"start. Parameter set: {para_set_num}/ {total_para_size}; {hyper_para.__str__()}")

        start_time = time.perf_counter()

        feature_based_result_dict: dict = {}
        series_frame_num_score_log_mtx_dict_dict: dict = {}
        try:
            for series_name, series_sub_dir in input_series_name_sub_dir_dict.items():
                print(f"working on series_name: {series_name}")

                frame_num_node_id_coord_dict_dict = read_series_data(raw_3d_data_dir + series_sub_dir, series_name)


                return_series, final_result_list, frame_num_score_log_mtx_dict = cell_tracking_core_flow(series_name,
                                                                                      frame_num_node_id_coord_dict_dict,
                                                                                      hyper_para
                                                                                      )
                feature_based_result_dict[series_name] = final_result_list
                series_frame_num_score_log_mtx_dict_dict[series_name] = frame_num_score_log_mtx_dict




        except Exception as e:
            time.sleep(1)
            print()
            traceback.print_exc()
            print(f"series_name {series_name}. para {para_set_num}.  hyper_para: {hyper_para.__str__()}")
            exit()




        hyper_para_str: str = str(hyper_para.weight_tuple) + \
                                    "MD(" + str(hyper_para.max_moving_distance) + ")_" + \
                                    "CL(" + str(hyper_para.coord_length_for_vector) + ")_" + \
                                    "AML(" + str(hyper_para.average_movement_step_length) + ")_" + \
                                    "MTL(" + str(hyper_para.minimum_track_length) + ")_" + \
                                    "DR(" +  str(hyper_para.discount_rate_per_layer) + ")_"


        result_file_name: str = py_file_name + "_hp" + str(hyperpara_idx+1).zfill(3) + "__" + hyper_para_str
        abs_file_path: str = individual_result_dir + result_file_name

        py_file_name: str = Path(__file__).name.replace(".py", "_")
        save_score_log_to_excel(series_frame_num_score_log_mtx_dict_dict, score_log_dir, py_file_name)


        execution_time = time.perf_counter() - start_time
        generate_txt_file(hyperpara_idx, hyper_para, execution_time, abs_file_path)


        print("save_track_dictionary: ", abs_file_path)

        print(f"Execution time: {np.round(execution_time, 4)} seconds")






def __________object_start_label():
    raise Exception("for labeling only")

CoordTuple = namedtuple("CoordTuple", "x y z")
WeightTuple = namedtuple("WeightTuple", "directional distance average_movement")


class DataStore(object):

    def __init__(self):
        profit_mtx_data_dict = {}


    @staticmethod
    def obtain_data_a(input_str: str):
        # return data_map_a_dict[input_str]
        print("Hi,this tutorial is about static class in python: " + input_str)



class ROUTING_STRATEGY_ENUM(enum.Enum):
    ALL_LAYER = 1
    ONE_LAYER = 2


class CUT_STRATEGY_ENUM(enum.Enum):
    DURING_ROUTING = 1
    AFTER_ROUTING = 2


class BOTH_CELL_BELOW_THRESHOLD_STRATEGY_ENUM(enum.Enum):
    SHARE = 1


class HyperPara():
    def __init__(self,
                 weight_tuple: tuple,
                 max_moving_distance: int,
                 coord_length_for_vector: int,
                 average_movement_step_length: int,
                 minimum_track_length: int,
                 discount_rate_per_layer: float):

        self.weight_tuple                     = weight_tuple
        self.max_moving_distance              = max_moving_distance
        self.coord_length_for_vector          = coord_length_for_vector
        self.average_movement_step_length     = average_movement_step_length
        self.minimum_track_length             = minimum_track_length
        self.discount_rate_per_layer          = discount_rate_per_layer


    def __str__(self, separator=""):
        return  f"weight_tuple: {self.weight_tuple}; {separator}" \
                f"max_moving_distance: {self.max_moving_distance}; {separator}" \
                f"coord_length_for_vector: {self.coord_length_for_vector}; {separator}" \
                f"average_movement_step_length: {self.average_movement_step_length}; {separator}" \
                f"minimum_track_length: {self.minimum_track_length}; {separator}" \
                f"discount_rate_per_layer: {self.discount_rate_per_layer}; {separator}"


class CellId():

    def __init__(self, start_frame_num: int, cell_idx: int):
        self.start_frame_num = start_frame_num
        self.cell_idx = cell_idx


    def str_short(self):
        return f"{self.start_frame_num}-{self.cell_idx};"


    def __str__(self):
        # return f"CellId(start_frame_num: {self.start_frame_num}; cell_idx: {self.cell_idx})"
        return f"CellId({self.start_frame_num}, {self.cell_idx})"


    def __eq__(self, other):
        if self.start_frame_num == other.start_frame_num and self.cell_idx == other.cell_idx:
            return True

        return False

    def __hash__(self):
        return hash((self.start_frame_num, self.cell_idx))


def compare_cell_id(cell_1, cell_2):
    if cell_1.start_frame_num == cell_2.start_frame_num:
        if cell_1.cell_idx < cell_2.cell_idx:           return -1
        elif cell_1.cell_idx > cell_2.cell_idx:         return 1
        else:                                           raise Exception("cell_1.node_idx == cell_2.node_idx")

    if cell_1.start_frame_num < cell_2.start_frame_num:            return -1
    elif cell_1.start_frame_num > cell_2.start_frame_num:          return 1


def __________flow_function_start_label():
    raise Exception("for labeling only")


def read_series_data(raw_3d_data_dir: str, series_name: str):
    image_path_list = sorted(list(paths.list_images(raw_3d_data_dir)))

    # construct image series list
    series_image_path_list_dict: dict = defaultdict(list)
    for idx, img_path in enumerate(image_path_list):
        series_image_path_list_dict[series_name].append(img_path)

    # sort
    for series_name, image_list in series_image_path_list_dict.items():
        series_image_path_list_dict[series_name] = sorted(image_list)


    threshold = 100  # if the volume of object is small than the threshold, remove the object
    total_size = len(series_image_path_list_dict)
    for idx, (series_name, img_path_list) in enumerate(series_image_path_list_dict.items()):

        # frame_num_coord_tuple_list_dict = {}
        frame_num_node_id_coord_dict_dict: dict = defaultdict(dict)
        total_frame = len(img_path_list)
        for frame_idx, img_path in enumerate(img_path_list):
            frame_num = frame_idx + 1
            print(f"\rseries: {idx + 1}/ {total_size}; frame: {frame_num}/{total_frame}", end='')

            img_stack = sitk.ReadImage(os.path.join(img_path))
            img_stack = sitk.GetArrayFromImage(img_stack)

            # get the intensity value and volume of each cell object
            img_stack_label, img_stack_cellvolume_counts = np.unique(img_stack, return_counts=True)

            # remove the small cell which volume is lower than threshold
            for l in range(len(img_stack_label)):
                if img_stack_cellvolume_counts[l] < threshold:
                    img_stack[img_stack == img_stack_label[l]] = 0
            labels = np.unique(img_stack)

            label_coord_tuple_dict = cell_center(img_stack)

            for label, coord_tuple in label_coord_tuple_dict.items():
                # key = str(frame_num) + ":" + str(label)
                # frame_num_coord_tuple_list_dict[key] = coord_tuple
                x, y, z = coord_tuple[0], coord_tuple[1], coord_tuple[2]

                frame_num_node_id_coord_dict_dict[frame_num][label] = CoordTuple(x, y, x)

        return frame_num_node_id_coord_dict_dict



def cell_center(seg_img):
    label_coord_tuple_dict = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_z, all_points_x, all_points_y = np.where(seg_img == label)
            avg_z = np.round(np.mean(all_points_z))
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))

            label_coord_tuple_dict[label] = (avg_z, avg_x, avg_y)

    return label_coord_tuple_dict



def cell_tracking_core_flow(series: str, frame_num_node_id_coord_dict_dict: dict, hyper_para: HyperPara):

    all_track_dict, score_log_mtx = execute_cell_tracking_task_bon(frame_num_node_id_coord_dict_dict,
                                                                   hyper_para
                                                                   )

    sorted_cell_id_key_list = sorted(list(all_track_dict.keys()), key=cmp_to_key(compare_cell_id))
    sorted_dict: dict = {}
    for sorted_key in sorted_cell_id_key_list:
        sorted_dict[sorted_key] = all_track_dict[sorted_key]
    all_track_dict = sorted_dict

    track_list_list: list = list(all_track_dict.values())

    code_validate_track_list(track_list_list)

    return series, track_list_list, score_log_mtx





def __________component_function_start_label():
    raise Exception("for labeling only")


def execute_cell_tracking_task_bon(frame_num_node_id_coord_dict_dict: dict, hyper_para: HyperPara):
    to_handle_cell_id_list: list = derive_initial_cell_id_list(frame_num_node_id_coord_dict_dict)

    frame_num_node_idx_occupation_list_list_dict = initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_id_coord_dict_dict)

    total_frame: int = max(frame_num_node_id_coord_dict_dict.keys())

    all_valid_cell_track_idx_dict: dict = {}        # [cell_id][frame_num] = node_idx
    all_valid_cell_track_score_dict: dict = {}       # [cell_id][frame_num] = probability

    score_log_mtx = init_score_log(frame_num_node_id_coord_dict_dict)

    while len(to_handle_cell_id_list) != 0:
        to_handle_cell_id_list.sort(key=cmp_to_key(compare_cell_id))

        to_handle_cell_id: CellId = to_handle_cell_id_list[0]

        print(f"{to_handle_cell_id.str_short()}")

        if to_handle_cell_id in all_valid_cell_track_idx_dict:
            frame_num_node_idx_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_list_list_dict, to_handle_cell_id, all_valid_cell_track_idx_dict[to_handle_cell_id])
            del all_valid_cell_track_idx_dict[to_handle_cell_id]
            del all_valid_cell_track_score_dict[to_handle_cell_id]


        is_new_cell: bool = (len(frame_num_node_idx_occupation_list_list_dict[to_handle_cell_id.start_frame_num][to_handle_cell_id.cell_idx]) == 0)
        if not is_new_cell:
            has_exist_data: bool = (to_handle_cell_id in all_valid_cell_track_idx_dict)
            if has_exist_data:
                frame_num_node_idx_occupation_list_list_dict = remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_list_list_dict, to_handle_cell_id, all_valid_cell_track_idx_dict[to_handle_cell_id])
                del all_valid_cell_track_idx_dict[to_handle_cell_id]
                del all_valid_cell_track_score_dict[to_handle_cell_id]

            to_handle_cell_id_list.remove(to_handle_cell_id)
            continue


        handling_cell_frame_num_track_idx_dict: dict = { to_handle_cell_id.start_frame_num: to_handle_cell_id.cell_idx}
        handling_cell_frame_num_score_dict: dict = {}

        redo_cell_id_set: set = set()

        second_frame_num: int = to_handle_cell_id.start_frame_num + 1
        for connect_to_frame_num in inclusive_range(second_frame_num, total_frame):
            total_node: int = len(frame_num_node_id_coord_dict_dict[connect_to_frame_num])
            is_connect_to_frame_has_no_cell = (total_node == 0)
            if is_connect_to_frame_has_no_cell:
                break


            node_id_coord_dict: dict = frame_num_node_id_coord_dict_dict[connect_to_frame_num]

            cut_threshold = 1000
            best_idx, best_prob, score_log_mtx, redo_cell_id_set = derive_best_node_idx_to_connect(to_handle_cell_id,
                                                                                                   handling_cell_frame_num_track_idx_dict,
                                                                                                   node_id_coord_dict,
                                                                                                   connect_to_frame_num,
                                                                                                   frame_num_node_idx_occupation_list_list_dict,
                                                                                                   all_valid_cell_track_idx_dict,
                                                                                                   all_valid_cell_track_score_dict,
                                                                                                   frame_num_node_id_coord_dict_dict,
                                                                                                   score_log_mtx,
                                                                                                   cut_threshold,
                                                                                                   redo_cell_id_set,
                                                                                                   hyper_para.weight_tuple,
                                                                                                   hyper_para.max_moving_distance,
                                                                                                   hyper_para.coord_length_for_vector,
                                                                                                   hyper_para.average_movement_step_length)




            has_valid_connection: bool = (best_prob != 0)
            if not has_valid_connection:
                break

            if connect_to_frame_num in handling_cell_frame_num_track_idx_dict:   raise Exception("code validation check")
            if connect_to_frame_num in handling_cell_frame_num_score_dict:       raise Exception("code validation check")

            handling_cell_frame_num_track_idx_dict[connect_to_frame_num] = best_idx
            handling_cell_frame_num_score_dict[connect_to_frame_num] = best_prob

        is_track_above_min_length: bool = (len(handling_cell_frame_num_track_idx_dict.keys()) > hyper_para.minimum_track_length)

        if not is_track_above_min_length:
            to_handle_cell_id_list.remove(to_handle_cell_id)
            continue


        # use final track to check if any redo cells occur
        for redo_cell_id in redo_cell_id_set:
            if redo_cell_id not in to_handle_cell_id_list:
                to_handle_cell_id_list.append(redo_cell_id)

        # add to occupation list
        frame_num_node_idx_occupation_list_list_dict = add_track_to_cell_occupation_list_list_dict(frame_num_node_idx_occupation_list_list_dict, to_handle_cell_id, handling_cell_frame_num_track_idx_dict)

        # add track to valid track list
        all_valid_cell_track_idx_dict[to_handle_cell_id] = handling_cell_frame_num_track_idx_dict
        all_valid_cell_track_score_dict[to_handle_cell_id] = handling_cell_frame_num_score_dict

        to_handle_cell_id_list.remove(to_handle_cell_id)

    print("----> finish")


    all_cell_id_track_list_dict: dict = defaultdict(list)
    for cell_id, handling_cell_frame_num_track_idx_dict in all_valid_cell_track_idx_dict.items():
        track_tuple_list: list = []
        previous_node_idx: int = -1
        for connect_to_frame_num, node_idx in handling_cell_frame_num_track_idx_dict.items():
            frame_idx: int = connect_to_frame_num - 1
            track_tuple_list.append((node_idx, frame_idx, previous_node_idx))
            previous_node_idx = node_idx

        all_cell_id_track_list_dict[cell_id] = track_tuple_list

    return all_cell_id_track_list_dict, score_log_mtx




def __________unit_function_start_label():
    raise Exception("for labeling only")

def generate_txt_file(hyperpara_idx: int, hyper_para: HyperPara, execution_time, abs_file_path: str):
    with open(abs_file_path + ".txt", 'w') as f:
        f.write(f"Execution time: {np.round(execution_time, 4)} seconds\n")
        f.write("hyper_para--- ID: " + str(hyperpara_idx + 1) + "; \n" + hyper_para.__str__(separator="\n"))



def derive_average_movement_score(to_handle_cell_id,
                                  current_frame_num,
                                  last_frame_node_coord,
                                  candidate_node_coord,
                                  step_length,
                                  handling_cell_frame_num_track_idx_dict,
                                  frame_num_node_id_coord_dict_dict,
                                  max_moving_distance: float):

    start_frame = current_frame_num - step_length
    start_frame = max(start_frame, 1, to_handle_cell_id.start_frame_num)
    end_frame = current_frame_num


    actual_step_length: int = (end_frame - start_frame) + 1
    if actual_step_length > 1:
        sum_distance: float = 0

        previous_node_idx: int = handling_cell_frame_num_track_idx_dict[start_frame]
        previous_coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[start_frame][previous_node_idx]
        for frame_num in inclusive_range(start_frame+1, end_frame):
            current_node_idx: int = handling_cell_frame_num_track_idx_dict[frame_num]
            current_coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[frame_num][current_node_idx]

            x_y_distance: float = ((current_coord_tuple.x - previous_coord_tuple.x)**2 + (current_coord_tuple.y - previous_coord_tuple.y)**2)**0.5
            distance: float = (x_y_distance**2 + (current_coord_tuple.z - previous_coord_tuple.z)**2)**0.5
            sum_distance += distance

            previous_coord_tuple = current_coord_tuple
        average_movement: float = sum_distance / actual_step_length

        new_candidate_node_distance: float = ((candidate_node_coord.x - last_frame_node_coord.x)**2 + (candidate_node_coord.y - last_frame_node_coord.y)**2)**0.5

        movement_difference: float = abs(average_movement - new_candidate_node_distance)

        if movement_difference > max_moving_distance:
            normalized_average_movement_score = 0
        else:
            normalized_average_movement_score: float = (max_moving_distance - movement_difference) / max_moving_distance

    else:
        normalized_average_movement_score = 0.5

    return normalized_average_movement_score



def derive_distance_score(current_node_coord, last_frame_node_coord, max_moving_distance):
    x_y_distance: float = ((current_node_coord.x - last_frame_node_coord.x)**2 + (current_node_coord.y - last_frame_node_coord.y)**2)**0.5

    x_y_z_distance: float = (x_y_distance**2 + (current_node_coord.z - last_frame_node_coord.z)**2)**0.5
    x_y_z_distance = np.round(x_y_z_distance, 4)

    if x_y_z_distance > max_moving_distance:
        distance_score = 0
    else:
        distance_score = (max_moving_distance - x_y_z_distance) / max_moving_distance

    return distance_score



def derive_directional_score(to_handle_cell_id, current_frame_num, last_frame_node_coord, candidate_node_coord,
                             coord_length_of_vector, handling_cell_frame_num_track_idx_dict, frame_num_node_id_coord_dict_dict):
    previous_coord_list = []
    start_frame = current_frame_num - coord_length_of_vector
    start_frame = max(start_frame, 1, to_handle_cell_id.start_frame_num)
    end_frame = current_frame_num

    coord_length: int = (end_frame - start_frame) + 1
    if coord_length > 1:
        for frame_num in inclusive_range(start_frame, end_frame):
            node_id: int = handling_cell_frame_num_track_idx_dict[frame_num]
            coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[frame_num][node_id]
            previous_coord_list.append(coord_tuple)
        previous_vec: CoordTuple = derive_vector_from_coord_list(previous_coord_list)

        new_candidate_vec: CoordTuple = derive_vector_from_coord_list([last_frame_node_coord, candidate_node_coord])

        degree_diff: float = derive_degree_diff_from_two_vectors(previous_vec, new_candidate_vec)

        directional_score: float = (cos(radians(degree_diff)) + 1) * 0.5  # +1 and *0.5 to shift up and make it stay between 1 and 0

    else:
        directional_score: float = 0.5


    return directional_score



def init_score_log(frame_num_node_id_coord_dict_dict):
    frame_num_score_log_mtx = {}

    start_frame_num = min(frame_num_node_id_coord_dict_dict.keys())
    end_frame_num = max(frame_num_node_id_coord_dict_dict.keys())
    for frame_num in range(start_frame_num, end_frame_num):
        num_node_id_coord_dict = frame_num_node_id_coord_dict_dict[frame_num]
        next_frame_num_node_id_coord_dict = frame_num_node_id_coord_dict_dict[frame_num+1]
        total_row = max(num_node_id_coord_dict.keys()) + 1
        total_col = max(next_frame_num_node_id_coord_dict) + 1
        row_list = []

        for row_idx in range(total_row):
            col_list = []
            for col_idx in range(total_col):
                col_list.append(0)
            row_list.append(col_list)
        frame_num_score_log_mtx[frame_num] = row_list

    return frame_num_score_log_mtx



def save_score_log_to_excel(series_frame_num_score_log_mtx_dict_dict: dict, excel_output_dir_path: str, file_name_prefix: str = ""):
    import pandas as pd

    for series, frame_num_score_log_mtx_dict in series_frame_num_score_log_mtx_dict_dict.items():
        file_name: str = file_name_prefix + f"series_{series}.xlsx"
        filepath = excel_output_dir_path + file_name;
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter') #pip install xlsxwriter

        for frame_num, prof_matrix in frame_num_score_log_mtx_dict.items():
            frame_data_array: np.arrays = frame_num_score_log_mtx_dict[frame_num]

            df = pd.DataFrame (frame_data_array)

            sheet_name: str = "frame_1" if frame_num == 1 else str(frame_num)

            df.to_excel(writer, sheet_name=sheet_name, index=True)

            workbook = writer.book
            cell_format = workbook.add_format({'text_wrap': True})        # to make \n work

            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('A:BO', cell_format=cell_format)

            for col_idx in range(40):
                if col_idx == 0:    col_width = 3
                else:               col_width = 25
                worksheet.set_column(col_idx, col_idx, col_width)

        writer.save()



def derive_vector_from_coord_list(coord_tuple_list: list):
    if len(coord_tuple_list) == 1:
        raise Exception("code validation check")

    start_vec: CoordTuple = coord_tuple_list[0]
    end_vec: CoordTuple = coord_tuple_list[-1]

    reversed_y_coord = -(end_vec.y - start_vec.y) # reverse y because coord raw data of y-axis is reversed
    z_coord = end_vec.z - start_vec.z
    final_coord_tuple = CoordTuple(end_vec.x - start_vec.x, reversed_y_coord, z_coord)

    return final_coord_tuple




def derive_initial_cell_id_list(frame_num_node_id_coord_dict_dict: dict):
    to_handle_cell_id_list: list = []
    for frame_num, node_id_coord_dict in frame_num_node_id_coord_dict_dict.items():
        for node_id in node_id_coord_dict.keys():
            cell_start_frame: int = frame_num
            to_handle_cell_id_list.append(CellId(cell_start_frame, node_id))

    return to_handle_cell_id_list




def derive_best_node_idx_to_connect(to_handle_cell_id,
                                    handling_cell_frame_num_track_idx_dict,
                                    node_id_coord_dict,
                                    connect_to_frame_num,
                                    frame_num_node_idx_occupation_list_list_dict,
                                    all_valid_cell_track_idx_dict,
                                    all_valid_cell_track_prob_dict,
                                    frame_num_node_id_coord_dict_dict,
                                    score_log_mtx,
                                    cut_threshold,
                                    redo_cell_id_set,
                                    weight_tuple: WeightTuple,
                                    max_moving_distance: int,
                                    coord_length_for_vector: int,
                                    average_movement_step_length: int):


    current_frame_num = connect_to_frame_num - 1
    round_to = 2

    current_frame_node_id: int = handling_cell_frame_num_track_idx_dict[current_frame_num]
    current_frame_node_coord: CoordTuple = frame_num_node_id_coord_dict_dict[current_frame_num][current_frame_node_id]

    best_prob: float = 0
    best_node_idx: int = None


    for candidate_node_id, candidate_node_coord in node_id_coord_dict.items():
        final_score: float = 0

        is_use_directional_score: bool = (weight_tuple.directional > 0)
        if is_use_directional_score:
            directional_score: float = derive_directional_score(to_handle_cell_id,
                                                                current_frame_num,
                                                                current_frame_node_coord,
                                                                candidate_node_coord,
                                                                coord_length_for_vector,
                                                                handling_cell_frame_num_track_idx_dict,
                                                                frame_num_node_id_coord_dict_dict)

            weighted_directional_score = np.round(weight_tuple.directional * directional_score, round_to)

            if math.isnan(weighted_directional_score):
                weighted_directional_score = 0



            final_score += weighted_directional_score


        is_use_distance_score: bool = (weight_tuple.distance > 0)
        if is_use_distance_score:
            distance_score = derive_distance_score(candidate_node_coord, current_frame_node_coord, max_moving_distance)

            weighted_distance_score = np.round(weight_tuple.distance * distance_score, round_to)
            final_score += weighted_distance_score


        is_use_avg_movement_score: bool = (weight_tuple.average_movement > 0)
        if is_use_avg_movement_score:
            avg_movement_score = derive_average_movement_score(to_handle_cell_id,
                                                               current_frame_num,
                                                               current_frame_node_coord,
                                                               candidate_node_coord,
                                                               average_movement_step_length,
                                                               handling_cell_frame_num_track_idx_dict,
                                                               frame_num_node_id_coord_dict_dict,
                                                               max_moving_distance)
            weighted_avg_mov_score = np.round(weight_tuple.average_movement * avg_movement_score, round_to)
            final_score += weighted_avg_mov_score


        final_score = np.round(final_score, round_to)

        score_log_mtx[current_frame_num][current_frame_node_id][candidate_node_id] = final_score

    return best_node_idx, best_prob, score_log_mtx, redo_cell_id_set



def initiate_frame_num_node_idx_cell_id_occupation_list_list_dict(frame_num_node_id_coord_dict_dict: dict):
    frame_num_node_idx_occupation_list_list_dict: dict = defaultdict(dict)

    for frame_num, node_id_coord_dict in frame_num_node_id_coord_dict_dict.items():
        for node_id in node_id_coord_dict.keys():
            frame_num_node_idx_occupation_list_list_dict[frame_num][node_id] = []

    return frame_num_node_idx_occupation_list_list_dict



def remove_track_from_cell_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, frame_num_cell_track_idx_dict: dict):
    for frame_num, node_idx in frame_num_cell_track_idx_dict.items():
        if cell_id not in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx]:
            raise Exception("code validation")

        frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx].remove(cell_id)

    return frame_num_node_idx_occupation_tuple_vec_dict



def add_track_to_cell_occupation_list_list_dict(frame_num_node_idx_occupation_tuple_vec_dict: dict, cell_id, frame_num_cell_track_idx_dict: dict):
    for frame_num, node_idx in frame_num_cell_track_idx_dict.items():
        if cell_id in frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx]:
            raise Exception("code validation")

        frame_num_node_idx_occupation_tuple_vec_dict[frame_num][node_idx].append(cell_id)


    return frame_num_node_idx_occupation_tuple_vec_dict




def __________general_function_start_label():
    raise Exception("for labeling only")



def derive_degree_diff_from_two_vectors(vector_coord_tuple_1: CoordTuple, vector_coord_tuple_2: CoordTuple): #These can also be four parameters instead of two arrays
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

        angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        angle_between((1, 0, 0), (1, 0, 0))
        0.0
        angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793
    """

    vec1_unit = unit_vector(vector_coord_tuple_1)
    vec2_unit = unit_vector(vector_coord_tuple_2)

    diff_radius: float = np.arccos(np.clip(np.dot(vec1_unit, vec2_unit), -1.0, 1.0))
    diff_degree: float = degrees(diff_radius)

    if diff_degree < 0:
        diff_degree = abs(diff_degree)

    return diff_degree



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)




def derive_degree_from_vector(vector_coord_tuple_1: CoordTuple): #These can also be four parameters instead of two arrays
    default_zero_degree_vector = CoordTuple(0, 1)

    dot = default_zero_degree_vector.x * vector_coord_tuple_1.x + default_zero_degree_vector.y * vector_coord_tuple_1.y      # dot product
    det = default_zero_degree_vector.y * vector_coord_tuple_1.x - default_zero_degree_vector.x * vector_coord_tuple_1.y      # determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    angle = degrees(angle)

    if angle < 0:
        angle += 360

    return angle



def inclusive_range(start: int, end: int):
    return range(start, end+1)



def dev_print(*arg):
    print(*arg)



def save_data_dict_to_file(data_dict: dict, save_file_abs_path: str):
    if not os.path.exists(save_file_abs_path):
        with open(save_file_abs_path, 'w'):
            pass
    pickle_out = open(save_file_abs_path, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()



def __________code_validation_function_start_label():
    raise Exception("for labeling only")



def code_validate_track_list(track_tuple_list_list: list):

    for track_tuple_list in track_tuple_list_list:
        frame_idx = track_tuple_list[0][1]

        for track_tuple in track_tuple_list[1:]:
            next_frame_idx = track_tuple[1]

            if next_frame_idx != (frame_idx + 1):
                raise Exception(track_tuple_list[0], next_frame_idx, track_tuple_list)

            frame_idx = next_frame_idx




if __name__ == '__main__':
    main()