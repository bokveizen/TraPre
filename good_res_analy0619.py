import pandas as pd
import numpy as np

df = pd.read_csv('res_analy/res_analy_1281.csv')
fol = [2, 14, 150, 326, 656, 783, 805, 884, 892, 942, 1033]
single_lc = [10, 26, 37, 41, 57, 73, 78, 86, 91, 96, 160, 197, 215, 219, 222, 230, 283, 330, 396, 484, 495, 572, 986]
fol_lc = [195, 356, 700, 896]
double_lc = [401, 640, 653, 748, 871, 1191]

fol_df = [df.iloc[i] for i in fol]
slc_df = [df.iloc[i] for i in single_lc]
flc_df = [df.iloc[i] for i in fol_lc]
dlc_df = [df.iloc[i] for i in double_lc]


def er_1s_mean_cal(df):
    er_1s_list = [i[1] for i in df]
    return np.mean(er_1s_list)


def er_3s_mean_cal(df):
    er_3s_list = [i[2] for i in df]
    return np.mean(er_3s_list)


print(er_1s_mean_cal(fol_df), er_3s_mean_cal(fol_df))
print(er_1s_mean_cal(slc_df), er_3s_mean_cal(slc_df))
print(er_1s_mean_cal(flc_df), er_3s_mean_cal(flc_df))
print(er_1s_mean_cal(dlc_df), er_3s_mean_cal(dlc_df))