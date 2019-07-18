from rels import *

# Vehicle list dividing and saving
df = pd.read_csv('trajectories-0400-0415.csv')
vList = []
currentVehicleID = 0
total_len = len(df)
current_index = 0
for i in df.iterrows():
    current_index += 1
    print(current_index, '/', total_len, "Current length of vList = ", len(vList))
    indexVehicleID = int(i[1]['Vehicle_ID'])
    if indexVehicleID > currentVehicleID:
        currentVehicleID = indexVehicleID
        vList.append([i[0], indexVehicleID])
saveFile = open("v_entry_list_FULL.bin", "wb")
pickle.dump(vList, saveFile)
saveFile.close()

# Data pre-process
# print(min(df['Lane_ID']), max(df['Lane_ID'])) # 1, 7
# timeBase = 1113433135300  # timeBase = min(df['Global_Time'])
# df['Global_Time'] = df['Global_Time'].map(lambda x: (x - timeBase) / 100)
# globalXBase = 6042593  # globalXBase = min(df['Global_X'])
# df['Global_X'] = df['Global_X'].map(lambda x: x - globalXBase)
# globalYBase = 2133053  # globalYBase = min(df['Global_Y'])
# df['Global_Y'] = df['Global_Y'].map(lambda x: x - globalYBase)