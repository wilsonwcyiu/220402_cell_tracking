import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas
import math


def main():
    distance_traveled_per_frame_micron_threshold: float = 2.4
    data_size_percentage_to_keep: int = 100

    project_folder_path: str = 'D:/program_source_code/220402_cell_tracking/220402_cell_tracking/main_evaluation_260101/'
    data_path: str = project_folder_path + 'data/'

    output_folder: str = data_path + "ttest_result/"
    measurement_file_path: str = data_path + '260112c_all_pkl_track_result_measurements.csv'
    df = pandas.read_csv(measurement_file_path)
    pkl_file_name_list: list[str] = [
                                    # 'gt_results_dict.pkl', 
                                    'kuan_tracks_allseries_unet.pkl', 
                                    'delta_results_dict.pkl', 
                                    'hungarian_results_dict.pkl', 
                                    'viterbi_results_dict_adj2.pkl',
                                    "Viterbi-like(Multi)__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl"
                                    ]

    file_name_to_method_name_dict: dict = {}
    file_name_to_method_name_dict["kuan_tracks_allseries_unet.pkl"] = "KDE"
    file_name_to_method_name_dict["delta_results_dict.pkl"] = "DELTA"
    file_name_to_method_name_dict["hungarian_results_dict.pkl"] = "Hungarian"
    file_name_to_method_name_dict["viterbi_results_dict_adj2.pkl"] = "Basic Viterbi"
    file_name_to_method_name_dict["Viterbi-like(Multi)__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl"] = "Extended Viterbi"


    measurement_input_list: list[MeasurementInput] = []
    measurement_input_list.append(MeasurementInput("Net Displacement", "net_displacement_microns"))
    measurement_input_list.append(MeasurementInput("Meandering index", "meandering_index_microns"))
    measurement_input_list.append(MeasurementInput("Mean speed", "mean_speed_microns"))

    fig, axes = plt.subplots(
        nrows=5,
        ncols=3,
        figsize=(40, 50),  # 512 x 512 pixels at dpi=100
        dpi=100
    )


    file_order_count: int = 1
    significant_single_data_str: str = ""
    fig_pos_x: int = 0
    for pkl_file_name in pkl_file_name_list:
        # print("pkl_file_name: ", pkl_file_name)
        method_name: str = file_name_to_method_name_dict[pkl_file_name]

        filtered_df = df[df['pkl_file_name'] == pkl_file_name]
        filtered_df = filtered_df[filtered_df["distance_traveled_per_frame_microns"] > distance_traveled_per_frame_micron_threshold]
        filtered_plus_df = filtered_df[filtered_df['cell_type'] == 'plus']
        filtered_minus_df = filtered_df[filtered_df['cell_type'] == 'minus']

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
            # print("measurement_type/ len(dataset_1)/ len(dataset_2)/ sig_label: ", measurement_type_fixed_length, "/ ", len(dataset_1), "/ ", len(dataset_2), "/ ", sig_label)
            

            fig_pos_y: int = 0
            if measurement_type == "Net Displacement": fig_pos_y = 0
            elif measurement_type == "Meandering index": fig_pos_y = 1
            elif measurement_type == "Mean speed": fig_pos_y = 2
            else: raise Exception("program error") 

            # tmp = axes[fig_pos_x, fig_pos_y]
            # print("dsfg", type(axes[fig_pos_x, fig_pos_y]))
            create_ttest_plot(axes, fig_pos_x, fig_pos_y, data_list, label_list, method_name, measurement_type, sig_label)

            # plt.tight_layout()
            # plt.show()
            
            output_pkl_file_name: str = None
            if pkl_file_name == "Viterbi-like(Multi)__viterbi_adjust4f_a_hp182__R(ALL)_M(0.89)_MIN(5)_CT(0.48)_ADJ(NO)_CS(D)_BB(S).pkl":
                output_pkl_file_name = "Viterbi Extended"
            else:
                output_pkl_file_name = pkl_file_name[0:-4]


            myd88_wt_np = np.array(myd88_wt)
            myd88_ko_np = np.array(myd88_ko)
            myd88_wt_mean = np.mean(myd88_wt_np).round(decimals=2)
            myd88_wt_std = np.std(myd88_wt_np).round(decimals=2)
            myd88_ko_mean = np.mean(myd88_ko_np).round(decimals=2)
            myd88_ko_std = np.std(myd88_ko_np).round(decimals=2)
            print(method_name, measurement_type, myd88_wt_mean, myd88_wt_std, myd88_ko_mean, myd88_ko_std)


            file_order_count += 1

        significant_single_data_str += ") "
        
        fig_pos_x += 1

        print("--")

    file_save_path: str = output_folder + "combined_plot"
    plt.savefig(file_save_path)



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



def create_ttest_plot(axis_plot, sub_plot_x, sub_plot_y, data_list, label_list, method_name, measurement_type, sig_label):

    dataset_1 = data_list[0]
    dataset_2 = data_list[1]

    # 4. 绘图
    # plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # 绘制散点箱线图
    palette_color_list: list[str] = ["#CFEBFD", "#FFFFD8"] 
    box_plot = sns.boxplot(data=data_list, width=0.8, palette=palette_color_list, ax=axis_plot[sub_plot_x, sub_plot_y])
    sns.stripplot(data=data_list, color='black', size=6, jitter=0.2, alpha=0.7, ax=axis_plot[sub_plot_x, sub_plot_y])

    # 添加显著性标记
    y_max = max(np.max(dataset_1), np.max(dataset_2)) * 1.1
    x1, x2 = 0, 1

    axis_plot[sub_plot_x, sub_plot_y].plot([x1, x1, x2, x2], [y_max, y_max+0.5, y_max+0.5, y_max], lw=1.5, color='black')

    # 设置标签和标题
    if sub_plot_x == 0:
        axis_plot[sub_plot_x, sub_plot_y].set_title(measurement_type + "\n\n" + sig_label, fontsize=32, weight='bold')
    else:
        axis_plot[sub_plot_x, sub_plot_y].set_title(sig_label, fontsize=32, weight='bold')
    


    axis_plot[sub_plot_x, sub_plot_y].set_xticks([0, 1], label_list, fontsize=32, weight='bold')

    y_axis_text: str = ""
    if measurement_type == "Net Displacement":
        y_axis_text = method_name + "\n\n"
    
    y_axis_text += measurement_type + " (μm/min)"

    axis_plot[sub_plot_x, sub_plot_y].set_ylabel(y_axis_text, fontsize=32, weight='bold')

    # axis_plot[sub_plot_x, sub_plot_y].set_yticks(axis='y', fontsize=32)
    axis_plot[sub_plot_x, sub_plot_y].tick_params(axis='y', which='major', labelsize=32)

    # ymin, ymax = plt.ylim()
    # plt.ylim(ymin, max(ymax, y_max + 1.5))

    # return plt



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