from rels import *

df = raw_df
test_set = range(len(vEntryList))
model_x_path = 'models/model_500_nosmx_06270529.h5'
model_y_path = 'models/model_500_nosmy_06241635.h5'
model_x = load_model(model_x_path)
model_y = load_model(model_y_path)
for i in test_set:
    # Data proc.
    end_point = min(vEntryList[i + 1][0], vEntryList[i][0] + 200)
    raw_data = pd.DataFrame(df[vEntryList[i][0]:end_point])
    test_df = df_process(raw_data)
    print("NO.", i, ":", "test DF loading and processing OK.")
    # smoothing and cutting
    cols_to_smooth = [col for col in test_df.columns if 'Local' in col]
    sm_start = 15
    sm_end = test_df.shape[0] - 15
    sm_len = 5
    # for col in cols_to_smooth:
    #     test_df = df_smoother(test_df, col, sm_start, sm_end, sm_len)
    test_df = test_df[sm_start:sm_end]
    test_df['final_X_pre'] = test_df['Local_X_pre']
    test_df['final_Y_pre'] = test_df['Local_Y_pre']
    test_df = test_df.drop(columns=['Local_X_pre', 'Local_Y_pre'])
    print("NO.", i, ":", "test data smoothing OK.")
    test_values = test_df.values
    gt_x = []
    gt_y = []
    pre_x = []
    pre_y = []
    for pre_col_sel in range(2):
        test = test_values
        pre_col_x = pre_col_sel  # 1 for x-pre, 0 for y-pre
        if pre_col_x:
            print("NO.", i, ":", "X data prediction starts.")
        else:
            print("NO.", i, ":", "Y data prediction starts.")
        pre_col_index = -1 - pre_col_x
        test_x, test_y = test[:, :-2], test[:, pre_col_index]
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        if pre_col_x:
            model = model_x
        else:
            model = model_y
        predict_y = model.predict(test_x)
        predict_y = np.reshape(predict_y, (predict_y.size,))
        v_init_x = test_df['Local_X_dif1_-1'].values[0]
        v_init_y = test_df['Local_Y_dif1_-1'].values[0]
        pos_init_x = test_df['Local_X'].values[0]
        pos_init_y = test_df['Local_Y'].values[0]
        pos_init = pos_init_x if pre_col_x else pos_init_y
        v_init = v_init_x if pre_col_x else v_init_y
        stab_len = 100
        predict_y[:stab_len] = test_y[:stab_len]
        predict_y = [sum(predict_y[:i]) + v_init for i in range(len(predict_y))]
        test_y = [sum(test_y[:i]) + v_init for i in range(len(test_y))]
        predict_y = [sum(predict_y[:i]) + pos_init for i in range(len(predict_y))]
        test_y = [sum(test_y[:i]) + pos_init for i in range(len(test_y))]
        if pre_col_x:
            gt_x = test_y
            pre_x = predict_y
        else:
            gt_y = test_y
            pre_y = predict_y
        if pre_col_x:
            print("NO.", i, ":", "X prediction OK.")
        else:
            print("NO.", i, ":", "Y prediction OK.")
    res_data = [gt_x, pre_x, gt_y, pre_y]
    res_data = np.transpose(res_data).tolist()
    res_col_name = ['gt_x', 'pre_x', 'gt_y', 'pre_y']
    res_df = pd.DataFrame(columns=res_col_name, data=res_data)
    res_csv_name = 'res_' + str(i) + '.csv'
    res_path_name = 'res_0701_500nosm'
    if not os.path.exists(res_path_name):
        os.mkdir(res_path_name)
    res_df.to_csv(os.path.join(res_path_name, res_csv_name))
    print("NO.", i, ":", "result saving OK.")
