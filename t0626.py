import pickle

data_bin_filename = 'df_concat_res_0605_100.bin'
data_bin_file = open(data_bin_filename, 'rb')
df_to_train = pickle.load(data_bin_file)
data_bin_file.close()

vel_rates = []
for i in range(65055):
    data = df_to_train.iloc[i]
    vel_x = data['Local_X_dif1_-1']
    vel_y = data['Local_Y_dif1_-1']
    vel_x = 0.0001 if vel_x == 0 else vel_x
    vel_y = 0.0001 if vel_y == 0 else vel_y
    vel_rate = abs(vel_x / vel_y)
    vel_rates.append(vel_rate)

df_to_train['vel_rate'] = vel_rates
saveFile = open('df_concat_res_0625_with_vel_rate_100.bin', 'wb')
pickle.dump(df_to_train, saveFile)
saveFile.close()