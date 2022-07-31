#import cc3d
from collections import defaultdict
from os.path import dirname, abspath

import numpy as np
import os
import tifffile
from imutils import paths
import SimpleITK as sitk
import json

def derive_series_name(image_file_abs_path: str, raw_folder_abs_path: str):
    parent_dir_str = dirname(abspath(image_file_abs_path))

    series_name = parent_dir_str.replace(raw_folder_abs_path.replace("/", "\\"), "")
    series_name = series_name.replace(" ", "_").replace("\\", "__")

    return series_name


def cell_center(seg_img):
    label_coord_tuple_dict = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_z,all_points_x,all_points_y = np.where(seg_img==label)
            avg_z = np.round(np.mean(all_points_z))
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            # label_coord_tuple_dict[label]=[avg_z,avg_x,avg_y]


            label_coord_tuple_dict[label] = (avg_z, avg_x, avg_y)

    return label_coord_tuple_dict

# raw_folder_path = "E://3D cell tracking//4 3D segmentation//0 Segmented data//222//"
raw_folder_path = "D:/viterbi linkage/dataset/3D raw data_seg data_find center coordinate/4 Segmentation dataset/"
                  # + "1 8layers mask data/20190621++2_8layers_M3a_Step98/"
image_path_list = sorted(list(paths.list_images(raw_folder_path)))

coord_output_dir = "D:/viterbi linkage/dataset/coord_data_3d/"
# print(len(image_path_list))


# saved_filtered_small_objects_folder_path = "E://3D cell tracking//4 3D segmentation//0 Segmented data//333//"


#construct image series list
series_image_path_list_dict: dict = defaultdict(list)
for idx, img_path in enumerate(image_path_list):
    series_name = derive_series_name(img_path, raw_folder_path)
    series_image_path_list_dict[series_name].append(img_path)

#sort
for series_name, image_list in series_image_path_list_dict.items():
    series_image_path_list_dict[series_name] = sorted(image_list)

# for series, image_list in series_image_list_dict.items():
#     print("fsfgs", series, image_list[0])
# exit()


threshold = 100   #if the volume of object is small than the threshold, remove the object
total_size = len(series_image_path_list_dict)
for idx, (series_name, img_path_list) in enumerate(series_image_path_list_dict.items()):

    frame_num_coord_tuple_list_dict = {}
    total_frame = len(img_path_list)
    for frame_idx, img_path in enumerate(img_path_list):
        frame_num = frame_idx + 1
        print(f"\rseries: {idx+1}/ {total_size}; frame: {frame_num}/{total_frame}", end='')

        # basename = os.path.basename(img_file)
        # img_stack = sitk.ReadImage(os.path.join(raw_folder_path, basename))
        # print("gerwgf", img_file)
        # print("asfgsd", parent_dir_str)
        # print("ergrgg", image_set_name)
        img_stack = sitk.ReadImage(os.path.join(img_path))
        img_stack = sitk.GetArrayFromImage(img_stack)

        # get the intensity value and volume of each cell object
        img_stack_label,img_stack_cellvolume_counts = np.unique(img_stack,return_counts=True)

        # remove the small cell which volume is lower than threshold
        for l in range(len(img_stack_label)):
            if img_stack_cellvolume_counts[l]<threshold:
                img_stack[img_stack==img_stack_label[l]]=0
        labels = np.unique(img_stack)

        label_coord_tuple_dict = cell_center(img_stack)

        for label, coord_tuple in label_coord_tuple_dict.items():
            key = str(frame_num) + ":" + str(label)
            frame_num_coord_tuple_list_dict[key] = coord_tuple

    # for frame_num, label_coord_tuple_list_dict in frame_num_coord_tuple_list_dict.items():
    #     for label, coord_tuple_list_dict in label_coord_tuple_list_dict.items():
    #         print("asdfas", frame_num, label, coord_tuple_list_dict)
    #
    # exit()
    json_string = json.dumps(frame_num_coord_tuple_list_dict)
    # print()
    # print(json_string)

    # with open(coord_output_dir + series_name + ".json", 'w') as outfile:
    #     json.dump(json_string, outfile)

    with open(coord_output_dir + series_name + ".json", 'w') as outfile:
        outfile.write(json_string)

    # with open(coord_output_dir + series_name + ".json") as json_file:
    #     data = json.load(json_file)
    #     print("gsgr", data)
    # exit()


