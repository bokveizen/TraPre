from rels import *

df = raw_df
test_set = range(len(vEntryList))
model_x_1s_path = 'models/model_directly_pos_dif_x_1s_06291218.h5'
model_y_1s_path = 'models/model_directly_pos_dif_y_1s_07010358.h5'
model_x_3s_path = 'models/model_directly_pos_dif_x_1s_06291218.h5'
model_y_3s_path = 'models/model_directly_pos_dif_y_1s_07010358.h5'
model_x_1s = load_model(model_x_1s_path)
model_y_1s = load_model(model_y_1s_path)
model_x_3s = load_model(model_x_3s_path)
model_y_3s = load_model(model_y_3s_path)
for i in test_set:
    # Data proc.
    end_point = vEntryList[i + 1][0]
    raw_data = pd.DataFrame(df[vEntryList[i][0]:end_point])
    test_df = df_process(raw_data)
    print("NO.", i, ":", "test DF loading and processing OK.")
    # smoothing and cutting
    cols_to_smooth = [col for col in test_df.columns if 'Local' in col]
    sm_start = 50
    sm_end = test_df.shape[0] - 50
    sm_len = 5
    for col in cols_to_smooth:
        test_df = df_smoother(test_df, col, sm_start, sm_end, sm_len)
    pos_X_dif_after_1s = []
    pos_Y_dif_after_1s = []
    pos_X_dif_after_3s = []
    pos_Y_dif_after_3s = []
    total_data_len = len(test_df)
    for j in range(total_data_len):
        current_data = test_df.iloc[j]
        current_X = current_data['Local_X']
        current_Y = current_data['Local_Y']
        if j + 10 in range(total_data_len):
            after_1s_data = test_df.iloc[j + 10]
            after_1s_X, after_1s_Y = after_1s_data['Local_X'], after_1s_data['Local_Y']
            pos_X_dif_after_1s.append(after_1s_X - current_X)
            pos_Y_dif_after_1s.append(after_1s_Y - current_Y)
        else:
            pos_X_dif_after_1s.append(0)
            pos_Y_dif_after_1s.append(0)
        if j + 30 in range(total_data_len):
            after_3s_data = test_df.iloc[j + 30]
            after_3s_X, after_3s_Y = after_3s_data['Local_X'], after_3s_data['Local_Y']
            pos_X_dif_after_3s.append(after_3s_X - current_X)
            pos_Y_dif_after_3s.append(after_3s_Y - current_Y)
        else:
            pos_X_dif_after_3s.append(0)
            pos_Y_dif_after_3s.append(0)
    test_df['X_dif_1s'] = pos_X_dif_after_1s
    test_df['Y_dif_1s'] = pos_Y_dif_after_1s
    test_df['X_dif_3s'] = pos_X_dif_after_3s
    test_df['Y_dif_3s'] = pos_Y_dif_after_3s
    test_df = test_df[sm_start:sm_end]
    test_df['final_X_pre'] = test_df['Local_X_pre']
    test_df['final_Y_pre'] = test_df['Local_Y_pre']
    test_df = test_df.drop(columns=['Local_X_pre', 'Local_Y_pre'])
    print("NO.", i, ":", "test data smoothing OK.")
    test_values = test_df.values
    gt_x_1s = []
    gt_y_1s = []
    gt_x_3s = []
    gt_y_3s = []
    pre_x_1s = []
    pre_y_1s = []
    pre_x_3s = []
    pre_y_3s = []
    for pre_col_sel in range(4):
        test = test_values
        # 0 for X 1s, 1 for Y 1s, 2 for X 3s, 3 for Y 3s
        if pre_col_sel == 0:
            print("NO.", i, ":", "X 1s data prediction starts.")
        elif pre_col_sel == 1:
            print("NO.", i, ":", "Y 1s data prediction starts.")
        elif pre_col_sel == 2:
            print("NO.", i, ":", "X 3s data prediction starts.")
        else:
            print("NO.", i, ":", "Y 3s data prediction starts.")
        pre_col_index = -6 + pre_col_sel
        test_x, test_y = test[:, :-6], test[:, pre_col_index]
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        if pre_col_sel == 0:
            model = model_x_1s
        elif pre_col_sel == 1:
            model = model_y_1s
        elif pre_col_sel == 2:
            model = model_x_3s
        else:
            model = model_y_3s
        predict_y = model.predict(test_x)
        predict_y = np.reshape(predict_y, (predict_y.size,))
        if pre_col_sel == 0:
            test_y = test_df['X_dif_1s']
            gt_x_1s = test_y
            pre_x_1s = predict_y
            print("NO.", i, ":", "X 1s prediction OK.")
        elif pre_col_sel == 1:
            test_y = test_df['Y_dif_1s']
            gt_y_1s = test_y
            pre_y_1s = predict_y
            print("NO.", i, ":", "Y 1s prediction OK.")
        elif pre_col_sel == 2:
            test_y = test_df['X_dif_3s']
            gt_x_3s = test_y
            pre_x_3s = predict_y
            print("NO.", i, ":", "X 3s prediction OK.")
        else:
            test_y = test_df['Y_dif_3s']
            gt_y_3s = test_y
            pre_y_3s = predict_y
            print("NO.", i, ":", "Y 3s prediction OK.")
    res_data = [gt_x_1s, pre_x_1s, gt_y_1s, pre_y_1s,
                gt_x_3s, pre_x_3s, gt_y_3s, pre_y_3s]
    res_data = np.transpose(res_data).tolist()
    res_col_name = ['gt_x_1s', 'pre_x_1s', 'gt_y_1s', 'pre_y_1s',
                'gt_x_3s', 'pre_x_3s', 'gt_y_3s', 'pre_y_3s']
    res_df = pd.DataFrame(columns=res_col_name, data=res_data)
    res_csv_name = 'res_' + str(i) + '.csv'
    res_path_name = 'res_0703_directly_1s'
    if not os.path.exists(res_path_name):
        os.mkdir(res_path_name)
    res_df.to_csv(os.path.join(res_path_name, res_csv_name))
    print("NO.", i, ":", "result saving OK.")
