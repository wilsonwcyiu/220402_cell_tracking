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
import random
from itertools import combinations
import pickle
import math
import pandas as pd


def main():
    print(f">> Execution started.")
    start_time = time.perf_counter()
    # date_str: str = datetime.now().strftime("%Y%m%d-%H%M%S")


    print(">> setup paths")
    project_folder_path: str = 'D:/program_source_code/220402_cell_tracking/220402_cell_tracking/main_evaluation_260101/'
    data_path: str = project_folder_path + 'data/'
    pkl_data_path: str = data_path + 'pkl_data/'
    output_file_path: str = data_path + '260112c_all_pkl_track_result_measurements.csv'

    segmentation_folder = data_path + 'segmentation_unet_seg/'

    pkl_file_name_list: list = [
                                "gt_results_dict.pkl",
                                "delta_results_dict.pkl",                                
                                "hungarian_results_dict.pkl",                                
                                "kuan_tracks_allseries_unet.pkl",
                                "viterbi_results_dict_adj2.pkl",
                                "Viterbi-like(Multi)__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl"
                                ]

    print(">> setup configuration")
    filter_out_track_length_lower_than: int = 16
    pixel_to_microns_ratios: float = 0.8791

    myd88_plus_series_list: list = ['S10', 'S11', 'S14', 'S15', 'S18', 'S19']
    myd88_minus_series_list: list = ['S12', 'S13', 'S16', 'S17', 'S20']



    print(">> retrieve all segmentation folder paths")
    all_segmented_filename_list: list[str] = listdir(segmentation_folder)
    all_segmented_filename_list.sort()


    track_data_list: list[TrackData] = []
    for pkl_file_name in pkl_file_name_list:
        print(">> working on file", pkl_file_name)

        series_viterbi_result_list_dict: dict[list] = open_track_dictionary(pkl_data_path + pkl_file_name)   # key: series_id

        print(">> filter related series")
        for series_id in list(series_viterbi_result_list_dict.keys()):
            if series_id not in myd88_plus_series_list and series_id not in myd88_minus_series_list:
                print("remove series ", series_id)
                del series_viterbi_result_list_dict[series_id] 


        print(">> filter result with track length lower than ", filter_out_track_length_lower_than)
        for series, result_list_list in series_viterbi_result_list_dict.items():
            result_list_list = filter_track_by_length(result_list_list, filter_out_track_length_lower_than)
            series_viterbi_result_list_dict[series] = result_list_list
            

        print(">> generate cell coordinate data from cell id")
        tmp_list: list[TrackData] = generate_cell_data_with_coordinate_info(pkl_file_name, 
                                                                                    series_viterbi_result_list_dict, 
                                                                                    segmentation_folder, 
                                                                                    all_segmented_filename_list
                                                                                    )
        track_data_list.extend(tmp_list)

        
        print(">> determine if cell type is plus or minus")
        for track_data in track_data_list:
            if track_data.series_num in myd88_plus_series_list:
                track_data.cell_type = "plus"
            elif track_data.series_num in myd88_minus_series_list:
                track_data.cell_type = "minus"
            else:
                raise Exception("program flow error")



    print(">> calculate track results")
    for track_data in track_data_list:
        # the distance from the start point(P1) of a track to the end point(PN) as below. 
        net_displacement_pixel: float = calculate_coord_distance(track_data.start_coord_tuple, track_data.end_coord_tuple)
        track_data.net_displacement_pixel = net_displacement_pixel      
        track_data.net_displacement_microns = net_displacement_pixel / pixel_to_microns_ratios
                
        # distance travelled per frame
        distance_traveled_per_frame_pixel: float = net_displacement_pixel / track_data.total_frame
        track_data.distance_traveled_per_frame_pixel = distance_traveled_per_frame_pixel
        track_data.distance_traveled_per_frame_microns = distance_traveled_per_frame_pixel / pixel_to_microns_ratios

        # Meandering index: Net displacement divided by all journey of cells(P1+P2+P3+…+PN)
        total_travel_pixel_distance_of_track: float = calculate_total_travel_pixel_distance(track_data.track_coord_tuple_list)
        track_data.total_travel_pixel_distance_of_track = total_travel_pixel_distance_of_track

        meandering_index_pixel: float = net_displacement_pixel / total_travel_pixel_distance_of_track
        track_data.meandering_index_pixel = meandering_index_pixel
        track_data.meandering_index_microns = meandering_index_pixel / pixel_to_microns_ratios

        # all journey of cells(P1+P2+P3+…+PN) divided by frames(N)
        mean_speed_pixel: float = total_travel_pixel_distance_of_track / track_data.total_frame
        track_data.mean_speed_pixel = mean_speed_pixel
        track_data.mean_speed_microns = mean_speed_pixel / pixel_to_microns_ratios
        



    print(">> track_data_list to dataframe")
    df = pd.DataFrame([t.to_dict() for t in track_data_list])


    print(">> generate csv")
    df.to_csv(output_file_path)


    for track_data in track_data_list:
        coord_tuple_list: list[tuple] = track_data.track_coord_tuple_list



    execution_time = time.perf_counter() - start_time
    print(f">> Execution completed. Execution time: {execution_time: 0.4f} seconds")



def main_test_case():
    
    # test case
    coord_tuple_list: list[tuple] = [(0, 0), (3, 4)]
    total_travel_pixel_distance: float = calculate_total_travel_pixel_distance(coord_tuple_list)
    print(total_travel_pixel_distance)

    
    # test case
    coord_tuple_list: list[tuple] = [(0, 0), (3, 4), (6, 8)]
    total_travel_pixel_distance: float = calculate_total_travel_pixel_distance(coord_tuple_list)
    print(total_travel_pixel_distance)



CoordTuple = namedtuple("CoordTuple", "x y")

class TrackData:
     
    # Constructor
    def __init__(self, pkl_file_name, series_num, track_id, track_coord_tuple_list):
        # raw input data
        self.pkl_file_name: str = pkl_file_name
        self.series_num: str = series_num
        self.track_id: str = track_id       # series no:x_coord:y_coord
        self.track_coord_tuple_list: list[tuple] = track_coord_tuple_list


        # summarized data
        self.total_frame: int = len(track_coord_tuple_list)
        self.start_coord_tuple: tuple = track_coord_tuple_list[0]
        self.end_coord_tuple: tuple = track_coord_tuple_list[-1]
        self.abs_y_diff = abs(self.end_coord_tuple[1] - self.start_coord_tuple[1])


        # derived data
        self.total_travel_pixel_distance_of_track = None
        self.cell_type: str = None                     # {plus, minus}

        self.net_displacement_pixel: float = None      # the distance from the start point(P1) of a track to the end point(PN) as below. 
        self.meandering_index_pixel: float = None      # Meandering index: Net displacement divided by all journey of cells(P1+P2+P3+…+PN)
        self.mean_speed_pixel: float = None
        self.distance_traveled_per_frame_pixel: float = None

        self.net_displacement_microns: float = None
        self.meandering_index_microns: float = None
        self.mean_speed_microns: float = None
        self.distance_traveled_per_frame_microns: float = None






    def to_dict(self):
        track_coord_list: list = []
        for track_coord_tuple in self.track_coord_tuple_list:
            text: str = "(" + str(track_coord_tuple.x) + "," + str(track_coord_tuple.y) + ")"
            track_coord_list.append(text)

        return {
            'pkl_file_name': self.pkl_file_name,
            'series_num': self.series_num,
            'cell_type': self.cell_type,
            'track_id': self.track_id,
            'total_frame': self.total_frame,
            'distance_traveled_per_frame_microns': self.distance_traveled_per_frame_microns,
            'net_displacement_microns': self.net_displacement_microns,
            'meandering_index_microns': self.meandering_index_microns,
            'mean_speed_microns': self.mean_speed_microns,
            'distance_traveled_per_frame_pixel': self.distance_traveled_per_frame_pixel,
            'net_displacement_pixel': self.net_displacement_pixel,
            'meandering_index_pixel': self.meandering_index_pixel,
            'mean_speed_pixel': self.mean_speed_pixel,
            'total_travel_pixel_distance_of_track': self.total_travel_pixel_distance_of_track,
            'start_coord_tuple': self.start_coord_tuple._asdict(),
            'end_coord_tuple': self.end_coord_tuple._asdict(),
            'abs_y_diff':self.abs_y_diff,
            'track_coord_tuple_list': track_coord_list
        }



def generate_cell_data_with_coordinate_info(pkl_file_name: str, series_viterbi_result_list_dict: dict, segmentation_folder: str, all_segmented_filename_list: list):
    track_data_list: list = []
    for series, result_list_list in series_viterbi_result_list_dict.items():

        result_list_list = sorted(result_list_list)

        segmented_filename_list: list = derive_segmented_filename_list_by_series(series, all_segmented_filename_list)

        frame_num_node_id_coord_dict_dict = derive_frame_num_node_idx_coord_list_dict(segmentation_folder, segmented_filename_list)

        # convert data from cell id to coordinates
        for idx, result_list in enumerate(result_list_list):
            print(f"\r{idx+1}/ {len(result_list_list)}; ", end='')
            coord_tuple_list = []
            for result_tuple in result_list:
                cell_idx = result_tuple[0]
                frame_num = result_tuple[1] + 1
                coord_tuple: CoordTuple = frame_num_node_id_coord_dict_dict[frame_num][cell_idx]
                coord_tuple_list.append(coord_tuple)


            first_cell_coord_in_track_tuple: tuple = coord_tuple_list[0]
            x_coord: int = first_cell_coord_in_track_tuple[0]
            y_coord: int = first_cell_coord_in_track_tuple[1]

            track_id: str = series + ":x" + str(x_coord) + ":y" + str(y_coord)

            track_data: TrackData = TrackData(pkl_file_name, series, track_id, coord_tuple_list)
            track_data_list.append(track_data)

    return track_data_list



def calculate_coord_distance(coord_1_tuple: tuple, coord_2_tuple: tuple):
    pixel_distance: float = math.dist(coord_1_tuple, coord_2_tuple)
    #print(f"The Euclidean distance is: {distance}")
    
    return pixel_distance


def calculate_total_travel_pixel_distance(track_coord_tuple_list: list[tuple]):
    total_travel_distance_of_track: float = 0

    second_last_coord_idx: int = len(track_coord_tuple_list) - 1
    for idx in range(0, second_last_coord_idx):
        current_coord_tuple: tuple = track_coord_tuple_list[idx]
        next_coord_tuple: tuple = track_coord_tuple_list[idx + 1]

        total_travel_distance_of_track += calculate_coord_distance(current_coord_tuple, next_coord_tuple)

    return total_travel_distance_of_track


def derive_segmented_filename_list_by_series(series: str, segmented_filename_list: list):
    result_segmented_filename_list: list = []

    for segmented_filename in segmented_filename_list:
        if series in segmented_filename:
            result_segmented_filename_list.append(segmented_filename)

    return result_segmented_filename_list



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