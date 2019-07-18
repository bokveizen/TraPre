from rels import *

df = raw_df
test_set = range(len(vEntryList))
model_path = 'models/model_turn_intention06271301.h5'
model = load_model(model_path)
for i in test_set:
    # Data proc.
    # end_point = min(vEntryList[i + 1][0], vEntryList[i][0] + 200)
    end_point = vEntryList[i + 1][0]
    raw_data = pd.DataFrame(df[vEntryList[i][0]:end_point])
    test_df = df_process(raw_data)
    print("NO.", i, ":", "test DF loading and processing OK.")
    # smoothing and cutting
    cols_to_smooth = [col for col in test_df.columns if 'Local' in col]
    sm_start = 15
    sm_end = test_df.shape[0] - 15
    sm_len = 5
    for col in cols_to_smooth:
        test_df = df_smoother(test_df, col, sm_start, sm_end, sm_len)
    test_df = test_df[sm_start:sm_end]
    test_df['final_X_pre'] = test_df['Local_X_pre']
    test_df['final_Y_pre'] = test_df['Local_Y_pre']
    test_df = test_df.drop(columns=['Local_X_pre', 'Local_Y_pre'])
    print("NO.", i, ":", "test data smoothing OK.")
    turn_intentions = []
    for ii in range(len(test_df) - 10):
        data = test_df.iloc[i]
        data_after_1s = test_df.iloc[i + 10]
        current_lane_id = data['Lane_ID']
        next_lane_id = data_after_1s['Lane_ID']
        turn_intention = next_lane_id - current_lane_id
        turn_intentions.append(turn_intention)
    for ii in range(10):
        turn_intentions.append(turn_intentions[-1])
    test_df['turn_intention'] = turn_intentions
    test_values = test_df.values
    test = test_values
    print("NO.", i, ":", "data prediction starts.")
    test_x, test_y = test[:, :-3], test[:, -1]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predict_y = model.predict(test_x)
    predict_y = np.reshape(predict_y, (predict_y.size,))
    gt = test_y
    pre = predict_y
    print("NO.", i, ":", "prediction OK.")
    res_data = [gt, pre]
    res_data = np.transpose(res_data).tolist()
    res_col_name = ['gt', 'pre']
    res_df = pd.DataFrame(columns=res_col_name, data=res_data)
    res_csv_name = 'res_' + str(i) + '.csv'
    res_path_name = 'res_0701_turn_intention'
    if not os.path.exists(res_path_name):
        os.mkdir(res_path_name)
    res_df.to_csv(os.path.join(res_path_name, res_csv_name))
    print("NO.", i, ":", "result saving OK.")
