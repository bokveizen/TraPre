from rels import *

df = raw_df
first = 1
final_df = df
train_set_len = 100  # len(vEntryList) - 100 == 1625
for i in range(train_set_len):
    # Data proc.
    raw_data = pd.DataFrame(df[vEntryList[i][0]:vEntryList[i + 1][0]])
    new_df = df_process(raw_data)

    # smoothing and cutting
    cols_to_smooth = [col for col in new_df.columns if 'Local' in col]
    sm_start = 50
    sm_end = new_df.shape[0] - 50
    sm_len = 5
    for col in cols_to_smooth:
        new_df = df_smoother(new_df, col, sm_start, sm_end, sm_len)
    pos_X_dif_after_1s = []
    pos_Y_dif_after_1s = []
    pos_X_dif_after_3s = []
    pos_Y_dif_after_3s = []
    total_data_len = len(new_df)
    for j in range(total_data_len):
        current_data = new_df.iloc[j]
        current_X = current_data['Local_X']
        current_Y = current_data['Local_Y']
        if j + 10 in range(total_data_len):
            after_1s_data = new_df.iloc[j + 10]
            after_1s_X, after_1s_Y = after_1s_data['Local_X'], after_1s_data['Local_Y']
            pos_X_dif_after_1s.append(after_1s_X - current_X)
            pos_Y_dif_after_1s.append(after_1s_Y - current_Y)
        else:
            pos_X_dif_after_1s.append(0)
            pos_Y_dif_after_1s.append(0)
        if j + 30 in range(total_data_len):
            after_3s_data = new_df.iloc[j + 30]
            after_3s_X, after_3s_Y = after_3s_data['Local_X'], after_3s_data['Local_Y']
            pos_X_dif_after_3s.append(after_3s_X - current_X)
            pos_Y_dif_after_3s.append(after_3s_Y - current_Y)
        else:
            pos_X_dif_after_3s.append(0)
            pos_Y_dif_after_3s.append(0)
    new_df['X_dif_1s'] = pos_X_dif_after_1s
    new_df['Y_dif_1s'] = pos_Y_dif_after_1s
    new_df['X_dif_3s'] = pos_X_dif_after_3s
    new_df['Y_dif_3s'] = pos_Y_dif_after_3s
    new_df = new_df[sm_start:sm_end]
    new_df['final_X_pre'] = new_df['Local_X_pre']
    new_df['final_Y_pre'] = new_df['Local_Y_pre']
    new_df = new_df.drop(columns=['Local_X_pre', 'Local_Y_pre'])
    if first:
        first = 0
        final_df = new_df
    else:
        final_df = final_df.append(new_df, ignore_index=True)
    print('No.', i + 1, 'Vehicle DF save OK.', i + 1, '/', train_set_len, final_df.shape)

saveFile = open('df_concat_res_0628_100_with_directly_1s_3s.bin', 'wb')
pickle.dump(final_df, saveFile)
saveFile.close()
