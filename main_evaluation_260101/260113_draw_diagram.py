
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas
import math
import ast

def main():
    sigificant_data_list: list[float] = []

    filter_the_first_x_longest_track: int = 30


    # distance_traveled_per_frame_micron_threshold: float = 1.6
    data_size_percentage_to_keep: int = 100

    project_folder_path: str = 'D:/program_source_code/220402_cell_tracking/220402_cell_tracking/main_evaluation_260101/'
    data_path: str = project_folder_path + 'data/'

    output_folder: str = data_path + "ttest_result/trajectory_graph/"
    measurement_file_path: str = data_path + '260112a_all_pkl_track_result_measurements.csv'
    df = pandas.read_csv(measurement_file_path)

    pkl_file_name_list: list[str] = [
                                    'gt_results_dict.pkl', 
                                    # 'delta_results_dict.pkl', 
                                    # 'hungarian_results_dict.pkl', 
                                    # 'kuan_tracks_allseries_unet.pkl', 
                                    # 'viterbi_results_dict_adj2.pkl',
                                    # "Viterbi-like(Multi)__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl"
                                    ]
    
    for pkl_file_name in pkl_file_name_list:
        print("pkl_file_name: ", pkl_file_name)

        for index, row in df.iterrows():
            track_id: str = row['track_id']
            track_str: str = track_id.replace(":", "-")
            coord_tuple_str: str  = row['track_coord_tuple_list']
            coord_tuple_str = coord_tuple_str.replace('\'', '')
            
            coord_tuple_list: list[str] = ast.literal_eval(coord_tuple_str)
            # print(coord_tuple_list)
            draw_diagram(coord_tuple_list, output_folder, track_str + ".png")









def draw_diagram(coord_tuple_list: list[tuple], abs_dir_path: str, file_name: str):
    # Split into x and y
    x = [p[0] for p in coord_tuple_list]
    y = [p[1] for p in coord_tuple_list]

    # Plot coord_tuple_list + lines
    plt.figure()
    plt.plot(x, y, marker='o', linewidth=1)  # line + coord_tuple_list

    # Labels
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Diagram with Connected Lines")

    sequence = np.arange(len(coord_tuple_list))
    scatter = plt.scatter(
        x, y,
        c=sequence,
        cmap='viridis',   # perceptually uniform
        s=60
    )

    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)

    cmap='viridis_r'

    # draw arrow
    for i in range(len(x) - 1):
        plt.arrow(
            x[i], y[i],
            x[i+1] - x[i], y[i+1] - y[i],
            length_includes_head=True,
            head_width=0.1,
            alpha=0.6
        )

    plt.savefig(abs_dir_path + file_name)
    plt.close


if __name__ == '__main__':
    main()