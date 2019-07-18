from rels import *

df = raw_df
first = 1
final_df = df
train_set_len = 1000  # len(vEntryList) - 100 == 1625
for i in range(train_set_len):
    # Data proc.
    raw_data = pd.DataFrame(df[vEntryList[i][0]:vEntryList[i + 1][0]])
    new_df = df_process(raw_data)

    # smoothing and cutting
    cols_to_smooth = [col for col in new_df.columns if 'Local' in col]
    sm_start = 15
    sm_end = new_df.shape[0] - 15
    sm_len = 5
    # for col in cols_to_smooth:
    #     new_df = df_smoother(new_df, col, sm_start, sm_end, sm_len)
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

saveFile = open('df_concat_res_0605_1000.bin', 'wb')
pickle.dump(final_df, saveFile)
saveFile.close()
