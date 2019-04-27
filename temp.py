# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# 按人和时间双排序
# 时间转换
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def time_transform(times):
    ts = time.mktime(time.strptime('{}'.format(times),
                                  "%Y/%m/%d : %H:%M:%S"))
    return ts

data = pd.read_csv("data/jdata/jdata_action.csv")
data = data.iloc[100000,:]

i = 0
length = len(data)
nun_user = None
temp1 = {}
buy_dauer = np.array([])

while i < length:
    if data.iloc[i, 0] != nun_user:
        nun_user = data.iloc[i, 0]
        temp1 = {}
    if not data.iloc[i, 1] in temp1.keys():
        if data.iloc[i, 4] in (1, 3, 5):
            temp1[data.iloc[i, 1]] = data.iloc[i, 2]
        elif data.iloc[i, 4] == 2:
            np.append(buy_dauer, 0)
    else:
        if data.iloc[i, 4] == 2:
            die_time = data.iloc[i, 2] - temp1[data.iloc[i, 1]]
            die_time = int(die_time/86400)
            np.append(buy_dauer, die_time)
            del temp1[data.iloc[i, 1]]

# plot
plt.hist(buy_dauer, bins=20, color='green')
plt.show()