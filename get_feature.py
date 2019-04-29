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
shop_path = "data/jdata/jdata_shop.csv"




class windows():
    
    def __init__(self, start_date, mid_date, end_date, subset = 100000):
        '''
        start_date：窗口开始时间
        mid_date：窗口中段时间
        end_date：窗口结束时间
        subset：取子样本数量
        '''
        self.subset = subset
        # 商品特征
        self.fets1 = ['sku_id', 'cate', 'brand', 'market_time', 'shop_id']
        # 店铺特征
        self.fets2 = ['shop_id', 'fans_num', 'vip_num', "shop_reg_tm",
                    'shop_score']
        # 用户特征
        self.fets3 = ["user_id", "age", "sex", "sex", "user_reg_tm",
                       "user_reg_tm",
                       "user_lv_cd", "city_level", "city_level", "province",
                       "province", "city", "city", "county"]
        
        
        self.start_date = self.time_transform(start_date)
        self.mid_date = self.time_transform(mid_date)
        self.end_date = self.time_transform(end_date)
        
        self.actions1 = self.get_actions(self.start_date,
                                    self.mid_date)
        self.actions2 = self.get_actions(self.mid_date,
                                    self.end_date)
        self.feats = self.get_action_feat()
        self.get_feature_product_shop()
        
    def time_transform(self, times):
        '''
        时间转换函数
        返回：把文本"%Y-%m-%d %H:%M:%S" 转换为时间戳
        '''
        
        if len(times) == 21:
            times = times[:-2]
            
        ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01

        return ts
    
    def get_action_feat(self):
        '''
        获取一个窗口内的主要特征
        包括目标变量、浏览时长和浏览次数
        '''
    
        dump_path = './cache/action_accumulate_%s_%s_%s.pkl' % (
                self.start_date,
                self.mid_date,
                self.end_date)
        
        if os.path.exists(dump_path):
            actions_feat = pickle.load(open(dump_path, "rb"))
            return actions_feat
        
        # 查询当前遍历用户
        nun_user = self.actions1.loc[self.actions1.index[0],
                                "user_id"]
        
        temp1 = {} # 初次浏览时间
        temp2 = {} # 末次浏览时间
        temp3 = defaultdict(lambda: 0) # 总浏览次数
        view_dauer = np.array([]) # 浏览时长
        view_times = np.array([], dtype = int) # 浏览次数
        user_id = np.array([], dtype = int) # 用户ID
        sku_ids = np.array([], dtype = int) # 商品ID
        
        # 遍历行为数据，查找特征
        for i in self.actions1.index:
            # 如果当前遍历用户改变，结算浏览时长，更新缓存数据
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
        
        
        # 构建返回的数据框
        actions_feat = pd.DataFrame({"user_id" : user_id,
                          "sku_id" : sku_ids,
                          "view_dauer" : view_dauer,
                          "view_times" : view_times})
        # 创建目标变量
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
    
    def get_actions(self, start_date, end_date):
        """
        获取时间段内的actions数据
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
    
                     
    def get_product_shop(self):
        
       
        # sku_id对应商品信息
        dump_path = './cache/basic_product.pkl'
        if os.path.exists(dump_path):
            product = pickle.load(open(dump_path, "rb"))
        else:
            product = pd.read_csv(product_path)
            product = product.dropna()
            product["market_time"] = product["market_time"].map(
                    lambda x: self.time_transform(x))
            
            product = product[self.fets1]
            pickle.dump(product, open(dump_path, 'wb'))
        
        ps_dict = product.set_index('sku_id').T.to_dict('list')
        ps_dict = defaultdict(lambda :np.full([len(self.fets1) - 1],
                                               np.nan), ps_dict)
        # shop_id对应店铺信息
        dump_path = './cache/basic_shop.pkl'
        if os.path.exists(dump_path):
            shop = pickle.load(open(dump_path, "rb"))
        else:
            shop = pd.read_csv(shop_path)
            shop = shop.dropna()
            shop["shop_reg_tm"] = shop["shop_reg_tm"].map(
                    lambda x: self.time_transform(x))
            shop = shop[self.fets2]
            pickle.dump(shop, open(dump_path, 'wb'))
            
        sp_dict = shop.set_index('shop_id').T.to_dict('list')
        sp_dict = defaultdict(lambda :np.full([len(self.fets2) - 1],
                                               np.nan), sp_dict)
        
        # user_id对应用户信息
        dump_path = './cache/basic_user.pkl'
        if os.path.exists(dump_path):
            user = pickle.load(open(dump_path, "rb"))
        else:
            user = pd.read_csv(user_path)
            user = user.dropna()
            user["user_reg_tm"] = user["user_reg_tm"].map(
                    lambda x: self.time_transform(x))
            age_df = pd.get_dummies(user["age"], prefix="age")
            sex_df = pd.get_dummies(user["sex"], prefix="sex")
            user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
            user = pd.concat([user[self.fets3], age_df, sex_df, user_lv_df], axis=1)
            self.fets3 = list(user)
            pickle.dump(user, open(dump_path, 'wb'))
            
        us_dict = user.set_index('user_id').T.to_dict('list')
        us_dict = defaultdict(lambda :np.full([len(self.fets3) - 1],
                                               np.nan), us_dict)
        
        return ps_dict, sp_dict, us_dict
    
    
    def get_feature_product_shop(self):
        
        ps_dict, sp_dict, us_dict = self.get_product_shop()
        
        for i in range(len(self.fets1) - 1):
            self.feats[self.fets1[i + 1]] = self.feats["sku_id"].map(lambda x:
                    ps_dict[x][i])
        
            
        for i in range(len(self.fets2) - 1):
            self.feats[self.fets2[i + 1]] = self.feats["sku_id"].map(lambda x:
                    sp_dict[ps_dict[x][-1]][i])
            
        for i in range(len(self.fets3) - 1):
            self.feats[self.fets3[i + 1]] = self.feats["user_id"].map(lambda x:
                    us_dict[x][i])
           
            
        self.feats.pop("")
        #self.feats = self.feats.dropna()
        
test = windows("2018-02-15 00:00:00",
                           "2018-02-20 00:00:00",
                           "2018-03-01 00:00:00")
test.feats.to_csv("expm.csv", index = False)

'''
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
'''