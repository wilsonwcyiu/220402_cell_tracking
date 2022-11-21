Feature-weighted algorithm for 3D cells 
========================


Background
-------------
This program calculates the weight of all the cell connections among each frame with the given parameters, in which Distance, Directional difference and Average Distance serves as the core determining factor of the final score.
The result of the connection probability will be presented in Excel format for further evaluation and usage.


Installation
-------------
Python 3.8 and the libraries shown below are required:
```sh
  $ pip install pickle5
  $ pip install imutils
  $ pip install SimpleITK  
```


Folder path settings
-------------
In the feature_based_gen_score_matrix_for_3d_data.py file, specify the path listed below:
folder_path = '{the root directory of the project which contains the raw image data and for the output results to be saved}'
raw_3d_data_dir = '{the child directory from folder_path to reach the parent image folder (if any)'

input_series_name_sub_dir_dict:
This dict should be provided with a key serves as a custom series name, the value contains the sub directory path that reaches the source images of that series


Hyper parameter settings
-------------
weight_tuple_list: the weight determines the influence of each feature in the algorithm, the sum of the weight should be always 1
max_moving_distance_list: the maximum distance that is considered valid. This is used to normalized the feature score.
coord_length_for_vector_list: this value determines the number of previous frames to be considered for directional vector calculation 
average_movement_step_length_list: the value determines the number of previous frames to be considered for average movement calculation
minimum_track_length_list: the minimum trajectory length to be considered valid


Execution
-------------
With the python set in the environment, execute python feature_weighted_gen_score_matrix_for_3d_data.py