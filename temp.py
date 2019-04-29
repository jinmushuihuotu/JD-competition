# -*- coding: utf-8 -*-
"""
Bahn
"""

import time
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

action_path = "newdata.csv"
comment_path = "data/jdata/jdata_comment.csv"
product_path = "data/jdata/jdata_product.csv"
user_path = "data/jdata/jdata_user.csv"


class windows():
    
    def __init__(self, start_date, mid_date, end_date, subset = 10000):
        self.subset = subset
        self.start_date = self.time_transform(start_date)
        self.mid_date = self.time_transform(mid_date)
        self.end_date = self.time_transform(end_date)
        
        self.actions1 = self.get_actions(self.start_date,
                                    self.mid_date)
        self.actions2 = self.get_actions(self.mid_date,
                                    self.end_date)
        self.feats = self.get_action_feat()
        
    def time_transform(self, times):
        # 转换时间的函数
        # 文本"%Y-%m-%d %H:%M:%S" 转换为时间戳
    
        if len(times) == 21:
            times = times[:-2]
            
        ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01

        return ts
    
    def get_action_feat(self):
    # 获取一个窗口内的累计特征
    # 目标变量、浏览时长和浏览次数
    
        dump_path = './cache/action_accumulate_%s_%s_%s.pkl' % (
                self.start_date,
                self.mid_date,
                self.end_date)
        
        if os.path.exists(dump_path):
            actions_feat = pickle.load(open(dump_path, "rb"))
            return actions_feat

        nun_user = self.actions1.loc[self.actions1.index[0],
                                "user_id"]
        temp1 = {} #初次浏览时间
        temp2 = {} #末次浏览时间
        temp3 = defaultdict(lambda: 0) #总浏览次数
        view_dauer = np.array([])
        view_times = np.array([], dtype = int)
        user_id = np.array([], dtype = int)
        sku_ids = np.array([], dtype = int)
        for i in self.actions1.index:
            if self.actions1.loc[i, "user_id"] != nun_user:
                skus = list(temp1.keys())
                for sku in skus:
                    dauer = temp2[sku] - temp1[sku]
                    dauer = int(dauer/86400) + 1    
                    view_dauer = np.append(view_dauer, dauer)
                    view_times = np.append(view_times, temp3[sku])
                    sku_ids = np.append(sku_ids, sku)
                    user_id = np.append(user_id, nun_user)   
                nun_user = self.actions1.loc[i, "user_id"]
                temp1 = {}
                temp2 = {}
                temp3 = defaultdict(lambda: 0)
            sku_id = self.actions1.loc[i, "sku_id"]
            temp3[sku_id] += 1
            temp2[sku_id] = self.actions1.loc[i, "action_time"]
            if sku_id not in temp1.keys():
                temp1[sku_id] = self.actions1.loc[i, "action_time"]
            
        actions_feat = pd.DataFrame({"user_id" : user_id,
                          "sku_id" : sku_ids,
                          "view_dauer" : view_dauer,
                          "view_times" : view_times})
        tar = np.array([], dtype = int)
        for i in actions_feat.index:
            has_2 = np.array(self.actions2[(self.actions2["user_id"] == 
                             actions_feat.loc[i, "user_id"]) &
                             (self.actions2["sku_id"] == 
                             actions_feat.loc[i, "sku_id"])]["type"])
            if 2 in has_2:
                tar = np.append(tar, 1)
            else:
                tar = np.append(tar, 0)
        actions_feat.insert(0, "tar", tar)
        pickle.dump(actions_feat, open(dump_path, 'wb'))
        
        return actions_feat

    def get_basic_user_feat(self):
        '''
            获取用户基本资料（年龄、性别，用户等级）
            所有特征['user_id', 'age', 'sex',
            'user_reg_tm', 'user_lv_cd', 'city_level',
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
    
    
    def get_basic_product_feat(self):
    
        dump_path = './cache/basic_product.pkl'
        if os.path.exists(dump_path):
            product = pickle.load(open(dump_path, "rb"))
        else:
            product = pd.read_csv(product_path)
            product["market_time"] = product["market_time"].map(
                    lambda x: int(self.time_transform(x)/86400) + 1)
            product = product[['sku_id', 'cate',
                                         'brand', 'market_time']]
            pickle.dump(product, open(dump_path, 'wb'))
        return product
    
    def get_actions(self, start_date, end_date):
        """
        :param start_date:
        :param end_date:
        :return: actions: pd.Dataframe
        """
        dump_path = './cache/all_action_%s_%s.pkl' % (start_date,
                                                      end_date)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, "rb"))
        else:
            actions = pd.read_csv(action_path).iloc[0:self.subset,:]
            actions = actions[(actions["action_time"] >= start_date) &
                              (actions["action_time"] < end_date)]
            pickle.dump(actions, open(dump_path, 'wb'))

        return actions

'''
def time_transform(times):
# 转换时间的函数
# 文本"%Y-%m-%d %H:%M:%S" 转换为时间戳
    
    if len(times) == 21:
        times = times[:-2]
            
    ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01
    return ts


数据排序部分
data = pd.read_csv(action_path)
data.iloc[:, 2] =  data.iloc[:, 2].map(time_transform)
data = data.sort_values(by = ["user_id",
                                "action_time"])
本函数用于绘制浏览-下单时间直方图
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


def get_action_feat(actions1, actions2):
    # 目标变量、浏览时长和浏览次数
    
    nun_user = actions1.loc[actions1.index[0], "user_id"]
    temp1 = {} #初次浏览时间
    temp2 = {} #末次浏览时间
    temp3 = defaultdict(lambda: 0) #总浏览次数
    view_dauer = np.array([])
    view_times = np.array([], dtype = int)
    user_id = np.array([], dtype = int)
    sku_ids = np.array([], dtype = int)
    for i in actions1.index:
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
        
    df = pd.DataFrame({"user_id" : user_id,
                      "sku_id" : sku_ids,
                      "view_dauer" : view_dauer,
                      "view_times" : view_times})
    tar = np.array([], dtype = int)
    for i in df.index:
        has_2 = np.array(actions2[(actions2["user_id"] == 
                         df.loc[i, "user_id"]) &
                         (actions2["sku_id"] == 
                         df.loc[i, "sku_id"])]["type"])
        if 2 in has_2:
            tar = np.append(tar, 1)
        else:
            tar = np.append(tar, 0)
    df.insert(0, "tar", tar)
    return df

def get_basic_user_feat():
  
        获取用户基本资料（年龄、性别，用户等级）
        所有特征['user_id', 'age', 'sex', 'user_reg_tm', 'user_lv_cd', 'city_level',
        'province', 'city', 'county']
    
    
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
        actions = pd.read_csv(action_path).iloc[0:5000000,:]
        actions = actions[(actions["action_time"] >= start_date) &
                          (actions["action_time"] < end_date)]
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
        #pickle.dump(actions_feat, open(dump_path, 'wb'))
    return actions_feat
test = get_accumulate_action_feat("2018-02-04 00:00:00",
                           "2018-02-05 12:00:00",
                           "2018-02-05 23:59:00")


buy_dauer = get_hin(data)
# plot
plt.hist(buy_dauer, bins=300, color='green')
plt.show()
'''