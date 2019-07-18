import pickle

data_bin_filename = 'df_concat_res_0605_100.bin'
data_bin_file = open(data_bin_filename, 'rb')
df_to_train = pickle.load(data_bin_file)
data_bin_file.close()

turn_intentions = []
for i in range(65045):
    data = df_to_train.iloc[i]
    data_after_1s = df_to_train.iloc[i + 10]
    current_lane_id = data['Lane_ID']
    next_lane_id = data_after_1s['Lane_ID']
    turn_intention = next_lane_id - current_lane_id
    turn_intentions.append(turn_intention)

for i in range(10):
    turn_intentions.append(turn_intentions[-1])

df_to_train['turn_intention'] = turn_intentions
saveFile = open('df_concat_res_0627_with_turn_intention_100.bin', 'wb')
pickle.dump(df_to_train, saveFile)
saveFile.close()
