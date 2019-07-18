import pandas as pd
import numpy as np

error_1s = []
error_3s = []
lateral_move_1s = []
lateral_move_3s = []
res_path = 'res0620_100ahead'
res_file_name = 'res0620_100ahead_only_x_error'


def er_cal(data):
    gt_x, pre_x, gt_y, pre_y = data[1:]
    er_x = abs(gt_x - pre_x)
    er_y = abs(gt_y - pre_y)
    er = (er_x * er_x + er_y * er_y) ** 0.5
    return er


def er_cal_with_rate(data_start, data_end, rate_x=0, rate_y=0, only_flag=0):
    gt_x_s, pre_x_s, gt_y_s, pre_y_s = data_start[1:]
    gt_x_e, pre_x_e, gt_y_e, pre_y_e = data_end[1:]
    pre_x_with_rate = rate_x * gt_x_s + (1.0 - rate_x) * pre_x_e
    pre_y_with_rate = rate_y * gt_y_s + (1.0 - rate_y) * pre_y_e
    er_x = abs(gt_x_e - pre_x_with_rate)
    er_y = abs(gt_y_e - pre_y_with_rate)
    if only_flag == 0:
        er = (er_x * er_x + er_y * er_y) ** 0.5
    elif only_flag == 1:
        er = er_x
    else:
        er = er_y
    return er


def lateral_move_cal(data_start, data_end):
    start_x = data_start[1]
    end_x = data_end[1]
    return abs(start_x - end_x)


total_count = 1707
type1_count = 0  # er_1s <= 0.2, er_3s <= 0.5
type2_count = 0  # er_1s <= 0.5, er_3s <= 1.5
type3_count = 0  # er_1s <= 1.0, er_3s <= 3.0
type4_count = 0  # er_1s <= 0.2
type5_count = 0  # er_3s <= 0.5
type6_count = 0  # er_1s <= 0.5
type7_count = 0  # er_3s <= 1.5

res_range = range(total_count)
for i in res_range:
    df = pd.read_csv(res_path + '/res_' + str(i) + '.csv')
    stab_len = 100
    data_start = df.iloc[stab_len]
    data_1s = df.iloc[stab_len + 10]
    data_3s = df.iloc[stab_len + 30]
    lm_1s = lateral_move_cal(data_start, data_1s)
    lm_3s = lateral_move_cal(data_start, data_3s)
    # er_1s = er_cal(data_1s)
    # er_3s = er_cal(data_3s)
    er_1s = er_cal_with_rate(data_start, data_1s, 0, 0, 1)
    er_3s = er_cal_with_rate(data_start, data_3s, 0, 0, 1)
    print(lm_1s, er_1s, lm_3s, er_3s)
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
    lateral_move_1s.append(lm_1s)
    lateral_move_3s.append(lm_3s)
    error_1s.append(er_1s)
    error_3s.append(er_3s)

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
res_analy_data = [lateral_move_1s, error_1s, lateral_move_3s, error_3s]
res_analy_data = np.transpose(res_analy_data).tolist()
res_analy_col_name = ['lat_move_1s', 'error_1s', 'lat_move_3s', 'error_3s']
res_analy_df = pd.DataFrame(columns=res_analy_col_name, data=res_analy_data)
res_analy_csv_name = res_file_name + '.csv'
res_analy_df.to_csv('res_analy/' + res_analy_csv_name)
