import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas
import math


def main():
    sigificant_data_list: list[float] = []

    filter_the_first_x_longest_track: int = 30


    # distance_traveled_per_frame_micron_threshold: float = 1.6
    data_size_percentage_to_keep: int = 100

    project_folder_path: str = 'D:/program_source_code/220402_cell_tracking/220402_cell_tracking/main_evaluation_260101/'
    data_path: str = project_folder_path + 'data/'

    output_folder: str = data_path + "ttest_result/"
    measurement_file_path: str = data_path + '260112b_all_pkl_track_result_measurements.csv'
    df = pandas.read_csv(measurement_file_path)

    pkl_file_name_list: list[str] = [
                                    'gt_results_dict.pkl', 
                                    'delta_results_dict.pkl', 
                                    'hungarian_results_dict.pkl', 
                                    'kuan_tracks_allseries_unet.pkl', 
                                    'viterbi_results_dict_adj2.pkl',
                                    "Viterbi-like(Multi)__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl"
                                    ]

    measurement_input_list: list[MeasurementInput] = []
    measurement_input_list.append(MeasurementInput("Net Displacement", "net_displacement_microns"))
    measurement_input_list.append(MeasurementInput("Meandering index", "meandering_index_microns"))
    measurement_input_list.append(MeasurementInput("Mean speed", "mean_speed_microns"))


    # print(">>>> distance_traveled_per_frame_micron_threshold: ", distance_traveled_per_frame_micron_threshold)

    significant_single_data_str: str = ""
    for pkl_file_name in pkl_file_name_list:
        print("pkl_file_name: ", pkl_file_name)
        filtered_df = df[df['pkl_file_name'] == pkl_file_name]
        # filtered_df = filtered_df[filtered_df["distance_traveled_per_frame_microns"] > distance_traveled_per_frame_micron_threshold]
        filtered_plus_df = filtered_df[filtered_df['cell_type'] == 'plus']
        filtered_minus_df = filtered_df[filtered_df['cell_type'] == 'minus']

        print(">> filter the first", filter_the_first_x_longest_track , "longest track")
        filtered_plus_df = filtered_plus_df.sort_values(by='net_displacement_microns', ascending=False).head(filter_the_first_x_longest_track)
        filtered_minus_df = filtered_minus_df.sort_values(by='net_displacement_microns', ascending=False).head(filter_the_first_x_longest_track)

        # print("filtered_plus_df: ", filtered_plus_df['net_displacement_microns'].round(2).to_list())
        # print("filtered_minus_df: ", filtered_minus_df['net_displacement_microns'].round(2).to_list())

        significant_single_data_str += "("
        for measurement_input in measurement_input_list:
            measurement_type: str = measurement_input.measurement_name
            filtered_plus_column_data = filtered_plus_df[measurement_input.measurement_column_name]
            filtered_plus_df_data_list: list[float] = reduce_list_data_size(filtered_plus_column_data.tolist(), data_size_percentage_to_keep)

            filtered_minus_column_data = filtered_minus_df[measurement_input.measurement_column_name]
            filtered_minus_data_list: list[float] = reduce_list_data_size(filtered_minus_column_data.tolist(), data_size_percentage_to_keep)


            myd88_wt = filtered_plus_df_data_list   # myd88+/+ 组
            myd88_ko = filtered_minus_data_list     # myd88-/- 组

            label_list = ['myd88+/+', 'myd88-/-']
            data_list = [myd88_wt, myd88_ko]

            dataset_1 = data_list[0]
            dataset_2 = data_list[1]

            # 2. 计算t检验的p值
            t_stat, p_value = stats.ttest_ind(myd88_wt, myd88_ko, equal_var=False)

            sig_label = get_significance(p_value)

            significant_single_data_str += sig_label.ljust(3) + " "
            measurement_type_fixed_length: str = measurement_type.ljust(15)
            print("measurement_type/ len(dataset_1)/ len(dataset_2)/ sig_label: ", measurement_type_fixed_length, "/ ", len(dataset_1), "/ ", len(dataset_2), "/ ", sig_label)
                        
            # plt = create_ttest_plot(data_list, label_list, measurement_type, sig_label)

            # plt.tight_layout()
            # plt.show()

            pkl_file_name = pkl_file_name.replace("(", "-")
            pkl_file_name = pkl_file_name.replace(")", "-")
            pkl_file_name = pkl_file_name.replace(".", "-")
            
            # if "(" in pkl_file_name:
            #     end_idx = pkl_file_name.index("(")
            #     pkl_file_name = pkl_file_name[0: end_idx]

            file_save_path: str = output_folder + pkl_file_name[0:-4] + "-" + measurement_type
            # plt.savefig(file_save_path)

        significant_single_data_str += ") "
        
        # print("--")



class MeasurementInput:

    # Constructor
    def __init__(self, measurement_name: str, measurement_column_name: str):
        # raw input data
        self.measurement_name: str = measurement_name
        self.measurement_column_name: str = measurement_column_name



def reduce_list_data_size(data_list: list, percentage_to_keep: int):
    original_length = len(data_list)
    items_to_keep = math.ceil((percentage_to_keep / 100.0) * original_length)

    return data_list[:items_to_keep]



def create_ttest_plot(data_list, label_list, measurement_type, sig_label):

    dataset_1 = data_list[0]
    dataset_2 = data_list[1]

    # # 2. 计算t检验的p值
    # t_stat, p_value = stats.ttest_ind(dataset_1, dataset_2, equal_var=False)

    # sig_label = get_significance(p_value)
    # measurement_type_fixed_length: str = measurement_type.ljust(15)
    # print("measurement_type/ len(dataset_1)/ len(dataset_2)/ sig_label", measurement_type_fixed_length, "/ ", len(dataset_1), "/ ", len(dataset_2), "/ ", sig_label)
    
    # 4. 绘图
    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")

    # 绘制散点箱线图
    box_plot = sns.boxplot(data=data_list, width=0.5, palette='Set2')
    sns.stripplot(data=data_list, color='black', size=1, jitter=0.2, alpha=0.7)

    # 添加显著性标记
    y_max = max(np.max(dataset_1), np.max(dataset_2)) * 1.1
    x1, x2 = 0, 1
    # plt.plot([x1, x1, x2, x2], [y_max, y_max+0.5, y_max+0.5, y_max], lw=1.5, color='black')
    plt.plot([x1, x1, x2, x2], [y_max, y_max+0.5, y_max+0.5, y_max], lw=1.5, color='black')
    plt.text((x1+x2)/2, y_max+0.8, sig_label, ha='center', va='bottom', fontsize=12)

    # 设置标签和标题
    plt.title("Distant neutrophils - " + measurement_type)

    plt.xticks([0, 1], label_list)
    plt.ylabel(measurement_type + " (μm/min)")

    ymin, ymax = plt.ylim()
    plt.ylim(ymin, max(ymax, y_max + 1.5))

    return plt



# 3. 定义显著性标记
def get_significance(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'
    
    
if __name__ == '__main__':
    main()