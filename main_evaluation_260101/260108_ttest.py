import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas

pixel_to_microns_ratios: float = 0.8791
save_folder: str = "G:/My Drive/leiden_university_course_materials/thesis/260101_thesis_followup/ttest_graphs/"
save_folder: str = "d:/ttest_graphs/"
file = "d:/260107_beta_all_pkl_track_result_data_massaged.csv"
df = pandas.read_csv(file)
print(df.head)

pkl_file_name_list: list[str] = ['delta_results_dict.pkl', 'gt_results_dict.pkl', 
                                'hungarian_results_dict.pkl', 'kuan_tracks_allseries_unet.pkl', 
                                'viterbi_results_dict_adj2.pkl']
# pkl_file_name: str = 'delta_results_dict.pkl'

for pkl_file_name in pkl_file_name_list:
    filtered_df = df[df['pkl_file_name'] == pkl_file_name]
    filtered_plus_df = df[df['Cell Type'] == 'plus']
    filtered_minus_df = df[df['Cell Type'] == 'minus']

    # measurement_type: str = "Net Displacement"
    # filtered_plus_column_data = filtered_plus_df['net_displacement_pixel'] / pixel_to_microns_ratios
    # filtered_plus_df_data_list: list[float] = filtered_plus_column_data.tolist()

    # filtered_minus_column_data = filtered_plus_df['net_displacement_pixel'] / pixel_to_microns_ratios
    # filtered_minus_data_list: list[float] = filtered_minus_column_data.tolist()


    # measurement_type: str = "Meandering index"
    # filtered_plus_column_data = filtered_plus_df['meandering_index_pixel'] / pixel_to_microns_ratios
    # filtered_plus_df_data_list: list[float] = filtered_plus_column_data.tolist()

    # filtered_minus_column_data = filtered_plus_df['meandering_index_pixel'] / pixel_to_microns_ratios
    # filtered_minus_data_list: list[float] = filtered_minus_column_data.tolist()

    
    measurement_type: str = "Mean speed"
    filtered_plus_column_data = filtered_plus_df['mean_speed_pixel'] / pixel_to_microns_ratios
    filtered_plus_df_data_list: list[float] = filtered_plus_column_data.tolist()

    filtered_minus_column_data = filtered_plus_df['mean_speed_pixel'] / pixel_to_microns_ratios
    filtered_minus_data_list: list[float] = filtered_minus_column_data.tolist()






    # # 1. 生成模拟数据（对应图中myd88+/+和myd88-/-组）
    # np.random.seed(42)
    # myd88_wt = np.random.normal(10, 3, 30)   # myd88+/+ 组
    # myd88_ko = np.random.normal(5, 2, 22)    # myd88-/- 组

    myd88_wt = filtered_plus_df_data_list   # myd88+/+ 组
    myd88_ko = filtered_minus_data_list    # myd88-/- 组



    data = [myd88_wt, myd88_ko]
    labels = ['myd88+/+', 'myd88-/-']

    # 2. 计算t检验的p值
    t_stat, p_value = stats.ttest_ind(myd88_wt, myd88_ko, equal_var=False)

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

    sig_label = get_significance(p_value)

    # 4. 绘图
    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")

    # 绘制散点箱线图
    box_plot = sns.boxplot(data=data, width=0.3, palette='Set2')
    sns.stripplot(data=data, color='black', size=4, jitter=0.2, alpha=0.7)

    # 添加显著性标记
    y_max = max(np.max(myd88_wt), np.max(myd88_ko)) * 1.1
    x1, x2 = 0, 1
    plt.plot([x1, x1, x2, x2], [y_max, y_max+0.5, y_max+0.5, y_max], lw=1.5, color='black')
    plt.text((x1+x2)/2, y_max+0.8, sig_label, ha='center', va='bottom', fontsize=12)

    # 设置标签和标题
    # plt.title('Distant neutrophils - Mean speed')
    plt.title("Distant neutrophils - " + measurement_type)

    plt.xticks([0, 1], labels)
    # plt.ylabel('Mean speed (μm/min)')  # 对应图D的纵坐标
    plt.ylabel(measurement_type + " (μm/min)")

    plt.tight_layout()
    # plt.show()

    file_save_path: str = save_folder + pkl_file_name[0:-4] + "-" + measurement_type
    plt.savefig(file_save_path)