from rels import *

# Dataframe loading
data_bin_filename = 'df_concat_res_0605_100.bin'
data_bin_file = open(data_bin_filename, 'rb')
df_to_train = pickle.load(data_bin_file)
data_bin_file.close()

# model fitting & saving / model loading
model_loading = 0  # 0 for training, 1 for loading
pre_col_x = 1  # 1 for x-pre, 0 for y-pre
cur_time = time.strftime("%m%d%H%M")
raw_values = df_to_train.values
train = raw_values
# test = raw_values[total_num - test_num:total_num]
pre_col_index = -1 - pre_col_x
train_x, train_y = train[:, :-2], train[:, pre_col_index]
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
# test_x, test_y = test[:, :-1], test[:, -1]
# test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
if not model_loading:
    model = model_fit(train_x, train_y, 1, 10)
    model_save_path = 'models'
    predict_col_name = 'x' if pre_col_x else 'y'
    model_name = 'model_' + predict_col_name + '_' + cur_time + '.h5'
    model.save(os.path.join(model_save_path, model_name))
else:
    model_x_path = ''
    model_y_path = ''
    if pre_col_x:
        model = load_model(model_x_path)
    else:
        model = load_model(model_y_path)
