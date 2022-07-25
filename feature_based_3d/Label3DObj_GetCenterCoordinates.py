#import cc3d
import numpy as np
import os
import tifffile
from imutils import paths
import SimpleITK as sitk

def cell_center(seg_img):
    results = {}
    for label in np.unique(seg_img):
        if label != 0:
            all_points_z,all_points_x,all_points_y = np.where(seg_img==label)
            avg_z = np.round(np.mean(all_points_z))
            avg_x = np.round(np.mean(all_points_x))
            avg_y = np.round(np.mean(all_points_y))
            results[label]=[avg_z,avg_x,avg_y]
    return results

# raw_folder_path = "E://3D cell tracking//4 3D segmentation//0 Segmented data//222//"
raw_folder_path = "D:/viterbi linkage/dataset/3D raw data_seg data_find center coordinate/4 Segmentation dataset/" + "1 8layers mask data/20190621++2_8layers_M3a_Step98/"
imagePaths = sorted(list(paths.list_images(raw_folder_path)))

# saved_filtered_small_objects_folder_path = "E://3D cell tracking//4 3D segmentation//0 Segmented data//333//"

threshold = 100   #if the volume of object is small than the threshold, remove the object
for img_file in imagePaths:
    #print("The time point is: ", i)
    basename = os.path.basename(img_file)
    img_stack = sitk.ReadImage(os.path.join(raw_folder_path,basename))
    img_stack = sitk.GetArrayFromImage(img_stack)
    
    # get the intensity value and volume of each cell object
    img_stack_label,img_stack_cellvolume_counts = np.unique(img_stack,return_counts=True)
    
    # remove the small cell which volume is lower than threshold
    for l in range(len(img_stack_label)):
        if img_stack_cellvolume_counts[l]<threshold:
            img_stack[img_stack==img_stack_label[l]]=0
    labels = np.unique(img_stack)
    
    # start_label=0                     # relabel all cells in order numbers
    # for label in labels:
    #     img_stack[img_stack==label]=start_label
    #     start_label = start_label+1
    # labels = np.unique(img_stack)
    """
    img_stack = sitk.GetImageFromArray(img_stack)    #  save the reordered images if necessary
    sitk.WriteImage(img_stack,os.path.join(saved_filtered_small_objects_folder_path,basename))
    
    img1 = sitk.ReadImage(os.path.join(saved_filtered_small_objects_folder_path,basename))
    img1 = sitk.GetArrayFromImage(img1)
    """
    centers = cell_center(img_stack)
    print(centers)


