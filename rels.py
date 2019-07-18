import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import os
import tensorflow as tf

# File loading
raw_df = pd.read_csv('trajectories-0400-0415.csv')

# Vehicle list loading
loadFile = open('v_entry_list_FULL.bin', 'rb')
vEntryList = pickle.load(loadFile)
loadFile.close()


# find vehicle ID in vEntryList and return the position in the list
def find_in_v_entry_list(v_id):
    for index in range(len(vEntryList)):
        if vEntryList[index][1] == v_id:
            return index
    return -1


# pick out the data of a vehicle
def single_vehicle_data_pick_out(i):
    return pd.DataFrame(raw_df[vEntryList[i][0]:vEntryList[i + 1][0]])


# position diff. operation
def pos_diff_col_generator(df, col_name, pre_flag):
    new_df = df
    # Diff.
    predict_col = col_name
    data_raw = new_df[predict_col]
    # Diff. 1
    for i in range(-1, 9):
        data_dif1 = data_raw.diff().shift(i).fillna(0)
        dif1_col_name = col_name + '_dif1_' + str(i)
        new_df[dif1_col_name] = data_dif1
    # Diff. 2
    data_dif1 = data_raw.diff().shift(-1).fillna(0)
    for i in range(9):
        data_dif2 = data_dif1.diff().shift(i).fillna(0)
        dif2_col_name = col_name + '_dif2_' + str(i)
        new_df[dif2_col_name] = data_dif2
    if pre_flag:
        # Prediction
        data_to_predict = data_dif1.diff().shift(-1).fillna(0)
        pre_col_name = col_name + '_pre'
        new_df[pre_col_name] = data_to_predict
    return new_df


# position diff. operation for none-host vehicles
def pos_diff_col_generator_for_none_host(df, col_name, pre_flag):
    new_df = df
    # Diff.
    predict_col = col_name
    data_raw = new_df[predict_col]
    # Diff. 1
    for i in [-1, 4, 9]:
        data_dif1 = data_raw.diff().shift(i).fillna(0)
        dif1_col_name = col_name + '_dif1_' + str(i)
        new_df[dif1_col_name] = data_dif1
    # Diff. 2
    data_dif1 = data_raw.diff().shift(-1).fillna(0)
    for i in [0, 4, 8]:
        data_dif2 = data_dif1.diff().shift(i).fillna(0)
        dif2_col_name = col_name + '_dif2_' + str(i)
        new_df[dif2_col_name] = data_dif2
    if pre_flag:
        # Prediction
        data_to_predict = data_dif1.diff().shift(-1).fillna(0)
        pre_col_name = col_name + '_pre'
        new_df[pre_col_name] = data_to_predict
    return new_df


# one hot transformer
def one_hot_transformer(df, col_name, value_set, name_set):
    new_df = df
    data_raw = new_df[col_name]
    for i in range(len(value_set)):
        one_hot_col_name = col_name + '_' + name_set[i]
        new_df[one_hot_col_name] = data_raw.map(lambda x: int(x == value_set[i]))
    # new_df = new_df.drop(columns=[col_name])
    return new_df


# preceding or following vehicle information processor
def pre_fol_proc(df, col_name):
    new_df = df
    exist_col_name = col_name + '_Exist'
    x_col_name = col_name + '_Local_X'
    y_col_name = col_name + '_Local_Y'
    v_class_col_name = col_name + '_v_Class'
    data_raw = new_df[col_name]
    x_data_temp = new_df['Local_X']
    y_data_temp = new_df['Local_Y']
    v_class_data_temp = new_df['v_Class']
    new_df[exist_col_name] = data_raw.map(lambda x: int(x != 0))
    new_df[x_col_name] = x_data_temp
    new_df[y_col_name] = y_data_temp
    new_df[v_class_col_name] = v_class_data_temp
    for index, row in new_df.iterrows():
        current_frame_id = row['Frame_ID']
        if row[col_name] != 0:
            list_index = find_in_v_entry_list(row[col_name])
            if list_index != -1:
                preceding_df = single_vehicle_data_pick_out(list_index)
                for ii, rr in preceding_df.iterrows():
                    if rr['Frame_ID'] == current_frame_id:
                        new_df.at[index, x_col_name] = rr['Local_X']
                        new_df.at[index, y_col_name] = rr['Local_Y']
                        new_df.at[index, v_class_col_name] = rr['v_Class']
    new_df = new_df.drop(columns=[col_name])
    # Proc for pre/fol's Local X & Y
    new_df = pos_diff_col_generator_for_none_host(new_df, x_col_name, 0)
    new_df = pos_diff_col_generator_for_none_host(new_df, y_col_name, 0)
    v_class_value_set = range(1, 4)
    v_class_name_set = ['motor', 'auto', 'truck']
    new_df = one_hot_transformer(new_df, v_class_col_name, v_class_value_set, v_class_name_set)
    return new_df


# dataframe proc for data of a vehicle
def df_process(df):
    # Useful information screening
    new_df = df[[
        'Frame_ID',
        'Local_X',
        'Local_Y',
        'v_Class',
        'Lane_ID',
        'Preceding',
        'Following',
        'Space_Headway',
        'Time_Headway'
    ]]
    new_df = new_df.reset_index(drop=True)

    # Proc for Local_X, Local_Y
    new_df = pos_diff_col_generator(new_df, 'Local_X', 1)
    new_df = pos_diff_col_generator(new_df, 'Local_Y', 1)

    # Proc for v_Class, Lane_ID
    v_class_value_set = range(1, 4)
    v_class_name_set = ['motor', 'auto', 'truck']
    new_df = one_hot_transformer(new_df, 'v_Class', v_class_value_set, v_class_name_set)
    lane_id_value_set = range(1, 8)
    lane_id_name_set = [str(i) for i in range(1, 8)]
    new_df = one_hot_transformer(new_df, 'Lane_ID', lane_id_value_set, lane_id_name_set)

    # Proc for Preceding & Following
    new_df = pre_fol_proc(new_df, 'Preceding')
    new_df = pre_fol_proc(new_df, 'Following')
    new_df = new_df.drop(columns=['Frame_ID'])
    return new_df


# dataframe smoother
def df_smoother(df, col_name, sm_start, sm_end, sm_len, times=1):
    new_df = df
    sm_col_name = col_name + '_smoothed'
    new_df[sm_col_name] = new_df[col_name]
    for t in range(times):
        data_temp = new_df[col_name]
        for i in range(sm_start, sm_end):
            new_df.at[i, sm_col_name] = np.mean(data_temp[i - sm_len:i + sm_len + 1])
    return new_df


def model_fit(train_x, train_y, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(512, input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='RMSProp')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
