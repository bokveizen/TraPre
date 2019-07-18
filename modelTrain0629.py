from rels import *

# Dataframe loading
data_bin_filename = 'df_concat_res_0628_100_with_directly_1s_3s.bin'
data_bin_file = open(data_bin_filename, 'rb')
df_to_train = pickle.load(data_bin_file)
data_bin_file.close()

# model fitting & saving / model loading
for i in range(4):
    # 0 for X 1s, 1 for Y 1s, 2 for X 3s, 3 for Y 3s
    cur_time = time.strftime("%m%d%H%M")
    raw_values = df_to_train.values
    train = raw_values
    pre_col_index = -6 + i
    train_x, train_y = train[:, :-6], train[:, pre_col_index]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    model = model_fit(train_x, train_y, 1000, 512)
    model_save_path = 'models'
    predict_col_names = ['x_1s', 'y_1s', 'x_3s', 'y_3s']
    predict_col_name = predict_col_names[i]
    model_name = 'model_directly_pos_dif_' + predict_col_name + '_' + cur_time + '.h5'
    model.save(os.path.join(model_save_path, model_name))
