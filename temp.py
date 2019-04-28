# -*- coding: utf-8 -*-
"""
Bahn
数据清洗，绘制直方图
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

action_path = "newdata.csv"
comment_path = "data/jdata/jdata_comment.csv"
product_path = "data/jdata/jdata_product.csv"
user_path = "data/jdata/jdata_user.csv"



def time_transform(times):
    if len(times) == 21:
        times = times[:-2]
            
    ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01
    return ts

data = pd.read_csv(action_path)
data.iloc[:, 2] =  data.iloc[:, 2].map(time_transform)
data = data.sort_values(by = ["user_id",
                                  "action_time"])
'''
本函数只用于绘制浏览-下单时间直方图
def get_hin(data):
    i = 0
    length = len(data)
    nun_user = None
    temp1 = {}
    buy_dauer = np.array([])
    
    while i < length:
        if data.loc[i, "user_id"] != nun_user:
            nun_user = data.loc[i, "user_id"]
            temp1 = {}
            
        if not data.loc[i, "sku_id"] in temp1.keys():
            if data.loc[i, "type"] in (1, 3, 5):
                temp1[data.loc[i, "sku_id"]] = data.loc[i, "action_time"]
            elif data.loc[i, "type"] == 2:
                buy_dauer = np.append(buy_dauer, 0)
                
        else:
            if data.loc[i, "type"] == 2:
                die_time = data.loc[i, "action_time"] - temp1[data.loc[i, "sku_id"]]
                die_time = int(die_time/86400) + 1
                buy_dauer = np.append(buy_dauer, die_time)
                del temp1[data.loc[i, "sku_id"]]
        i += 1
    return buy_dauer
'''

def get_action_feat(actions1, actions2):
    # 提取浏览时长
    i = 0
    length = len(actions1)
    nun_user = actions1.loc[0, "user_id"]
    temp1 = {} #初次浏览时间
    temp2 = {} #末次浏览时间
    temp3 = defaultdict(lambda: 0) #总浏览次数
    view_dauer = np.array([])
    view_times = np.array([], dtype = int)
    user_id = np.array([], dtype = int)
    sku_ids = np.array([], dtype = int)
    
    while i < length:
        if actions1.loc[i, "user_id"] != nun_user:
            skus = list(temp1.keys())
            for sku in skus:
                dauer = temp2[sku] - temp1[sku]
                dauer = int(dauer/86400) + 1    
                view_dauer = np.append(view_dauer, dauer)
                view_times = np.append(view_times, temp3[sku])
                sku_ids = np.append(sku_ids, sku)
                user_id = np.append(user_id, nun_user)   
            nun_user = actions1.loc[i, "user_id"]
            temp1 = {}
            temp2 = {}
            temp3 = defaultdict(lambda: 0)
        sku_id = actions1.loc[i, "sku_id"]
        temp3[sku_id] += 1
        temp2[sku_id] = actions1.loc[i, "action_time"]
        if sku_id not in temp1.keys():
            temp1[sku_id] = actions1.loc[i, "action_time"]
        i += 1
        
    df = pd.DataFrame({"user_id" : user_id,
                      "sku_id" : sku_ids,
                      "view_dauer" : view_dauer,
                      "view_times" : view_times})
    length = len(actions2)
    tar = np.array([], dtype = int)
    for i in range(length):
        if 2 in actions2[(actions2["user_id"] == 
                         df["user_id"][i]) &
                         (actions2["sku_id"] == 
                         df["sku_id"][i])]["type"]:
            tar[i] = 1
        else:
            tar[i] = 0
    df.insert(0, "tar", tar)
    return df

def get_basic_user_feat():
    '''
        获取用户基本资料（年龄、性别，用户等级）
        所有特征['user_id', 'age', 'sex', 'user_reg_tm', 'user_lv_cd', 'city_level',
        'province', 'city', 'county']
    '''
    
    dump_path = 'cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, "rb"))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"],
               prefix="user_lv_cd")
        user = pd.concat([user['user_id'],
            age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'wb'))

    return user


def get_basic_product_feat():

    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, "rb"))
    else:
        product = pd.read_csv(product_path)
        product["market_time"] = product["market_time"].map(
                lambda x: int(time_transform(x)/86400) + 1)
        product = product[['sku_id', 'cate',
                                     'brand', 'market_time']]
        pickle.dump(product, open(dump_path, 'wb'))
    return product

def get_actions(start_date, end_date):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    start_date = time_transform(start_date)
    end_date = time_transform(end_date)
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, "rb"))
    else:
        actions = pd.read_csv(action_path)
        actions = actions[(actions["time"] >= start_date) &
                          (actions["time"] < end_date)]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_action_feat(start_date, mid_date, end_date):
# 获取一个窗口内的累计特征
    dump_path = './cache/action_accumulate_%s_%s_%s.pkl' % (start_date,
                                                mid_date, end_date)
    if os.path.exists(dump_path):
        actions_feat = pickle.load(open(dump_path, "rb"))
    else:
        actions1 = get_actions(start_date, mid_date)
        actions2 = get_actions(mid_date, end_date)
        actions_feat = get_action_feat(actions1, actions2)
        pickle.dump(actions_feat, open(dump_path, 'wb'))
    return actions_feat


'''
buy_dauer = get_hin(data)
# plot
plt.hist(buy_dauer, bins=300, color='green')
plt.show()
'''