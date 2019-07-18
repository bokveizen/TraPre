import pandas as pd
import numpy as np

error_1s_x = []
error_1s_y = []
error_3s_x = []
error_3s_y = []
error_1s = []
error_3s = []
res_path = 'res_0703_directly_1s'
res_file_name = 'res_0703_directly_1s_1334'


def er_cal(data):
    gt_x_1s, pre_x_1s, gt_y_1s, pre_y_1s, gt_x_3s, pre_x_3s, gt_y_3s, pre_y_3s = data[1:]
    er_x_1s = abs(gt_x_1s - pre_x_1s)
    er_y_1s = abs(gt_y_1s - pre_y_1s)
    er_x_3s = abs(gt_x_3s - pre_x_3s*3)
    er_y_3s = abs(gt_y_3s - pre_y_3s*3)
    er_1s = (er_x_1s * er_x_1s + er_y_1s * er_y_1s) ** 0.5
    er_3s = (er_x_3s * er_x_3s + er_y_3s + er_y_3s) ** 0.5
    er = [er_x_1s, er_y_1s, er_x_3s, er_y_3s, er_1s, er_3s]
    return er


total_count = 0
type1_count = 0  # er_1s <= 0.2, er_3s <= 0.5
type2_count = 0  # er_1s <= 0.5, er_3s <= 1.5
type3_count = 0  # er_1s <= 1.0, er_3s <= 3.0
type4_count = 0  # er_1s <= 0.2
type5_count = 0  # er_3s <= 0.5
type6_count = 0  # er_1s <= 0.5
type7_count = 0  # er_3s <= 1.5
res_range = range(300)
for i in res_range:
    df = pd.read_csv(res_path + '/res_' + str(i) + '.csv')
    stab_len = 50
    sample_len = 10
    for ii in range(sample_len):
        data = df.iloc[stab_len + ii]
        er_list = er_cal(data)
        total_count += 1
        error_1s_x.append(er_list[0])
        error_1s_y.append(er_list[1])
        error_3s_x.append(er_list[2])
        error_3s_y.append(er_list[3])
        error_1s.append(er_list[4])
        error_3s.append(er_list[5])
        er_1s = er_list[4]
        er_3s = er_list[5]
        # res stat.
        if er_1s <= 0.2 and er_3s <= 0.5:
            type1_count += 1
        if er_1s <= 0.5 and er_3s <= 1.5:
            type2_count += 1
        if er_1s <= 1.0 and er_3s <= 3.0:
            type3_count += 1
        if er_1s <= 0.2:
            type4_count += 1
        if er_3s <= 0.5:
            type5_count += 1
        if er_1s <= 0.5:
            type6_count += 1
        if er_3s <= 1.5:
            type7_count += 1

print(res_path)
print('1s error average:', np.mean(error_1s))
print('1s error median:', np.median(error_1s))
print('1s error std:', np.std(error_1s))
print('3s error average:', np.mean(error_3s))
print('3s error median:', np.median(error_3s))
print('3s error std:', np.std(error_3s))
print('No. of vehicles whose 1s error <= 0.2 and 3s error <= 0.5:',
      type1_count, '{:.2%}'.format(type1_count / total_count))
print('No. of vehicles whose 1s error <= 0.5 and 3s error <= 1.5:',
      type2_count, '{:.2%}'.format(type2_count / total_count))
print('No. of vehicles whose 1s error <= 1.0 and 3s error <= 3.0:',
      type3_count, '{:.2%}'.format(type3_count / total_count))
print('No. of vehicles whose 1s error <= 0.2:',
      type4_count, '{:.2%}'.format(type4_count / total_count))
print('No. of vehicles whose 3s error <= 0.5:',
      type5_count, '{:.2%}'.format(type5_count / total_count))
print('No. of vehicles whose 1s error <= 0.5:',
      type6_count, '{:.2%}'.format(type6_count / total_count))
print('No. of vehicles whose 3s error <= 1.5:',
      type7_count, '{:.2%}'.format(type7_count / total_count))
res_analy_data = [error_1s_x, error_1s_y, error_3s_x, error_3s_y, error_1s, error_3s]
res_analy_data = np.transpose(res_analy_data).tolist()
res_analy_col_name = ['1x', '1y', '3x', '3y', '1', '3']
res_analy_df = pd.DataFrame(columns=res_analy_col_name, data=res_analy_data)
res_analy_csv_name = res_file_name + '.csv'
res_analy_df.to_csv('res_analy/' + res_analy_csv_name)
