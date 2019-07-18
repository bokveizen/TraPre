from rels import *

# Dataframe loading
data_bin_filename = 'df_concat_res_0625_with_vel_rate_100.bin'
data_bin_file = open(data_bin_filename, 'rb')
df_to_train = pickle.load(data_bin_file)
data_bin_file.close()

# model fitting & saving / model loading
model_loading = 0  # 0 for training, 1 for loading
cur_time = time.strftime("%m%d%H%M")
raw_values = df_to_train.values
train = raw_values
# test = raw_values[total_num - test_num:total_num]
train_x, train_y = train[:, :-1], train[:, -1]
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
# test_x, test_y = test[:, :-1], test[:, -1]
# test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
model = model_fit(train_x, train_y, 512, 1000)
model_save_path = 'models'
model_name = 'model_vel_rate' + cur_time + '.h5'
model.save(os.path.join(model_save_path, model_name))

