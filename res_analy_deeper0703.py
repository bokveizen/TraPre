import pandas as pd
import numpy as np

csv_data = pd.read_csv('res_analy/res0620_100ahead_with_rate0.5.csv')
er_1s_list = csv_data['error_1s']
er_3s_list = csv_data['error_3s']
lm_1s_list = csv_data['lat_move_1s']
lm_3s_list = csv_data['lat_move_3s']
er_1s_sort_list = sorted(range(len(er_1s_list)), key=lambda i: er_1s_list[i])
er_3s_sort_list = sorted(range(len(er_3s_list)), key=lambda i: er_3s_list[i])
bad_res_len = 171  # total no. = 1707
er_1s_bad_list = er_1s_sort_list[-bad_res_len:]
er_3s_bad_list = er_3s_sort_list[-bad_res_len:]
er_1s_list_without_bad_res = [er_1s_list[i] for i in range(len(er_1s_list)) if i not in er_3s_bad_list]
er_3s_list_without_bad_res = [er_3s_list[i] for i in range(len(er_3s_list)) if i not in er_3s_bad_list]
lm_1s_list_of_bad_res = [lm_1s_list[i] for i in range(len(lm_1s_list)) if i in er_3s_bad_list]
lm_3s_list_of_bad_res = [lm_3s_list[i] for i in range(len(lm_3s_list)) if i in er_3s_bad_list]
print('1s error average:', np.mean(er_1s_list_without_bad_res))
print('1s error median:', np.median(er_1s_list_without_bad_res))
print('1s error std:', np.std(er_1s_list_without_bad_res))
print('3s error average:', np.mean(er_3s_list_without_bad_res))
print('3s error median:', np.median(er_3s_list_without_bad_res))
print('3s error std:', np.std(er_3s_list_without_bad_res))
print('1s lateral move of the worst results average:', np.mean(lm_1s_list_of_bad_res))
print('1s lateral move of all results average:', np.mean(lm_1s_list))
print('3s lateral move of the worst results average:', np.mean(lm_3s_list_of_bad_res))
print('3s lateral move of all results average:', np.mean(lm_3s_list))