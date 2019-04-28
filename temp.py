# -*- coding: utf-8 -*-
"""
Bahn
数据清洗，绘制直方图
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_trans(data):
    def time_transform(times):
    if len(times) == 21:
        times = times[:-2]     
    ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d %H:%M:%S")) - 1517414400.0
    return ts

    data = pd.read_csv("data/jdata/jdata_action.csv")
    data.iloc[:, 2] =  data.iloc[:, 2].apply(time_transform)
    data = data.sort_values(by = ["user_id", "action_time"])
    return data
# 把时间换成时间戳，并按用户ID和时间戳排序

def get_hin(data):
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
                buy_dauer = np.append(buy_dauer, 0)
        
        else:
            if data.iloc[i, 4] == 2:
                die_time = data.iloc[i, 2] - temp1[data.iloc[i, 1]]
                die_time = int(die_time/86400) + 1
                buy_dauer = np.append(buy_dauer, die_time)
                del temp1[data.iloc[i, 1]]
        
        i += 1
    
    return buy_dauer

buy_dauer = get_hin(data)
# plot
plt.hist(buy_dauer, bins=300, color='green')
plt.show()