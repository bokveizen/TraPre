import pandas as pd
import numpy as np

gt_turn_intention = []
pre_turn_intention = []
res_path = 'res_0701_turn_intention'
res_file_name = 'res_0701_turn_intention'


def turn_intention_cal(data):
    if -0.5 <= data <= 0.5:
        return 0
    elif data > 0.5:
        return 1
    else:
        return -1


total_count = 0
type1_count = 0  # gt = pre = 0
type2_count = 0  # gt = pre = -1
type3_count = 0  # gt = pre = 1
type4_count = 0  # gt = 0, pre = 1 (0, 1)
type5_count = 0  # (0, -1)
type6_count = 0  # (1, 0)
type7_count = 0  # (1, -1)
type8_count = 0  # (-1, 0)
type9_count = 0  # (-1, 1)
res_range = range(1708)
for i in res_range:
    df = pd.read_csv(res_path + '/res_' + str(i) + '.csv')
    df_len = len(df)
    stab_len = 100
    sample_len = min(100, df_len - 120)
    for ii in range(sample_len):
        sample_data = df.iloc[stab_len + ii]
        raw_gt, raw_pre = sample_data['gt'], sample_data['pre']
        int_gt, int_pre = turn_intention_cal(raw_gt), turn_intention_cal(raw_pre)
        gt_turn_intention.append(int_gt)
        pre_turn_intention.append(int_pre)
        print(int_gt, int_pre)
        total_count += 1
        # res stat.
        if (int_gt, int_pre) == (0, 0):
            type1_count += 1
        elif (int_gt, int_pre) == (-1, -1):
            type2_count += 1
        elif (int_gt, int_pre) == (1, 1):
            type3_count += 1
        elif (int_gt, int_pre) == (0, 1):
            type4_count += 1
        elif (int_gt, int_pre) == (0, -1):
            type5_count += 1
        elif (int_gt, int_pre) == (1, 0):
            type6_count += 1
        elif (int_gt, int_pre) == (1, -1):
            type7_count += 1
        elif (int_gt, int_pre) == (-1, 0):
            type8_count += 1
        else:
            type9_count += 1
print(res_path)
print('No. of vehicles whose (gt, pre) = (0, 0):',
      type1_count, '{:.2%}'.format(type1_count / total_count))
print('No. of vehicles whose (gt, pre) = (-1, -1):',
      type2_count, '{:.2%}'.format(type2_count / total_count))
print('No. of vehicles whose (gt, pre) = (1, 1):',
      type3_count, '{:.2%}'.format(type3_count / total_count))
print('No. of vehicles whose (gt, pre) = (0, 1):',
      type4_count, '{:.2%}'.format(type4_count / total_count))
print('No. of vehicles whose (gt, pre) = (0, -1):',
      type5_count, '{:.2%}'.format(type5_count / total_count))
print('No. of vehicles whose (gt, pre) = (1, 0):',
      type6_count, '{:.2%}'.format(type6_count / total_count))
print('No. of vehicles whose (gt, pre) = (1, -1):',
      type7_count, '{:.2%}'.format(type7_count / total_count))
print('No. of vehicles whose (gt, pre) = (-1, 0):',
      type8_count, '{:.2%}'.format(type8_count / total_count))
print('No. of vehicles whose (gt, pre) = (-1, 1):',
      type9_count, '{:.2%}'.format(type9_count / total_count))
res_analy_data = [gt_turn_intention, pre_turn_intention]
res_analy_data = np.transpose(res_analy_data).tolist()
res_analy_col_name = ['gt', 'pre']
res_analy_df = pd.DataFrame(columns=res_analy_col_name, data=res_analy_data)
res_analy_csv_name = res_file_name + '.csv'
res_analy_df.to_csv('res_analy/' + res_analy_csv_name)
