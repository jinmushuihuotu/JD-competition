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
from collections import Counter

action_path = "data/jdata/jdata_action.csv"
comment_path = "data/jdata/jdata_comment.csv"
product_path = "data/jdata/jdata_product.csv"
user_path = "data/jdata/jdata_user.csv"
shop_path = "data/jdata/jdata_shop.csv"



class windows():
    
    def __init__(self, end_date,
                     y = 7, lange = 30,
                     subset = "all"):
        '''
        y = 取目标变量的间隔（天）
        lange = 窗口长度（天）
        end_date：窗口结束时间
        subset：取子样本数量
        '''
        time_start = time.time()
        
        if y < 0 or lange < 0:
            raise Exception("Error: y und lange must >= 0")
            
        self.__y = y
        self.__lange = 30
        self.subset = subset
        # 商品特征
        self.fets1 = ['sku_id', 'cate', 'brand', 'market_time', 'shop_id']
        # 店铺特征
        self.fets2 = ['shop_id', 'fans_num', 'vip_num', "shop_reg_tm",
                    'shop_score']
        # 用户特征
        self.fets3 = ["user_id", "user_reg_tm",
                       "user_lv_cd", "city_level",
                       "province", "city", "county"]
        
        self.end_date = self.time_transform(end_date)
        self.mid_date = self.end_date - 86400 * y
        self.start_date = self.end_date - 86400 * lange
        print("寻找行为数据")
        self.actions1 = self.get_actions(self.start_date,
                                        self.mid_date)
        self.actions2 = self.get_actions(self.mid_date,
                                        self.end_date)
        self.feats = self.get_action_feat()
        self.get_feature_product_shop()
        #if y != 0:
        #    self.lower_sample_data()
        
        time_end = time.time()
        print('time cost',time_end-time_start,'s')
        
    def lower_sample_data(self, percent = 1):
        '''percent:
        多数类别下采样的数量相对于少数类别样本数量的比例 
        '''
        print("进行欠采样")
        data1 = self.feats[self.feats['tar'] == 0] 
        
        # 将多数类别的样本放在data1 
        data0 = self.feats[self.feats['tar'] == 1]
        if len(data0) == 0:
            raise Exception("no row with tar = 1")
        # 将少数类别的样本放在data0
        index = np.random.randint(len(data1),
            size = percent * (len(self.feats) - len(data1)))
        # 随机给定下采样取出样本的序号
        lower_data1 = data1.iloc[list(index)]
        # 下采样
        self.feats = pd.concat([lower_data1, data0])

        
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
        print("正在构建基本特征")
        dump_path = './cache/action_acm_%s_%s_%s_%s.pkl' % (
                self.end_date,
                self.__y,
                self.__lange,
                self.subset)
        
        if os.path.exists(dump_path):
            actions_feat = pickle.load(open(dump_path, "rb"))
            return actions_feat
        
        print("查找特征，创建目标变量")
        tp_action = self.actions2[self.actions2["type"] == 2].copy()
        tp_action["type"] = 6
    
        actions_feat = self.actions1.append(tp_action).copy()
        
        dums = pd.get_dummies(actions_feat['type'], prefix = 'type')
        actions_feat = pd.concat([actions_feat[['user_id','sku_id']],
                                   dums], axis = 1)
        actions_feat = actions_feat.groupby(['user_id','sku_id'],
                                             as_index = False).sum()
        
        #actions_feat = actions_feat[(actions_feat["type_1"]!=0) | (actions_feat["type_6"]==0)]
        actions_feat = actions_feat[actions_feat["type_1"] > 0]
        
        actions_feat.loc[actions_feat["type_6"] > 0, "type_6"] = 1
        actions_feat.rename(columns={"type_6": "tar"},
                            inplace=True)
        
        # 构建返回的数据框
        #actions_feat = self.get_from_action_data(self.actions1)
        
        
        pickle.dump(actions_feat, open(dump_path, 'wb'))
        
        return actions_feat
    
    def get_actions(self, start_date, end_date):
        """
        获取时间段内的actions数据
        """
        dump_path = './cache/all_action_%s_%s_%s_%s.pkl' % (start_date,
                                                      end_date,
                                                      self.__y,
                                                      self.subset)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, "rb"))
            return actions
        
        elif os.path.exists('./cache/all_action.pkl'):
            actions = pickle.load(open('./cache/all_action.pkl',
                                     "rb"))#.iloc[0:1000000,:]
        else:
            actions = pd.read_csv(action_path)
            actions = actions.dropna()
            pickle.dump(actions, open('./cache/all_action.pkl', 'wb'))
            #actions = actions.iloc[0:1000000,:]
            
        actions = actions[(actions["action_time"] >= start_date) &
                                  (actions["action_time"] < end_date)]
        pickle.dump(actions, open(dump_path, 'wb'))

        return actions
    
                     
    def get_product_shop(self):
        
        null_dict = defaultdict(lambda :np.nan)
        
        dump_path = './cache/basic_product.pkl'
        if os.path.exists(dump_path):
            ps_dict = pickle.load(open(dump_path, "rb"))
        else:
            product = pd.read_csv(product_path)
            product = product.dropna()
            product["market_time"] = product["market_time"].map(
                    lambda x: self.time_transform(x))
            
            product = product[self.fets1]
            ps_dict = product.set_index('sku_id').to_dict('index')
            pickle.dump(ps_dict, open(dump_path, 'wb'))
        
        
        ps_dict = defaultdict(lambda :null_dict, ps_dict)
        
        
        dump_path = './cache/basic_shop.pkl'
        if os.path.exists(dump_path):
            sp_dict = pickle.load(open(dump_path, "rb"))
        else:
            shop = pd.read_csv(shop_path)
            shop = shop.dropna()
            shop["shop_reg_tm"] = shop["shop_reg_tm"].map(
                    lambda x: self.time_transform(x))
            shop = shop[self.fets2]
            sp_dict = shop.set_index('shop_id').to_dict('index')
            pickle.dump(sp_dict, open(dump_path, 'wb'))
            
        sp_dict = defaultdict(lambda :null_dict, sp_dict)
        
        dump_path = './cache/basic_user.pkl'
        if os.path.exists(dump_path):
            us_dict = pickle.load(open(dump_path, "rb"))
        else:
            user = pd.read_csv(user_path)
            user = user.dropna()
            user["user_reg_tm"] = user["user_reg_tm"].map(
                    lambda x: self.time_transform(x))
            age_df = pd.get_dummies(user["age"], prefix="age")
            sex_df = pd.get_dummies(user["sex"], prefix="sex")
            user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
            user = pd.concat([user[self.fets3], age_df, sex_df, user_lv_df], axis=1)
            self.fets3 = list(user.columns)
            us_dict = user.set_index('user_id').to_dict('index')
            pickle.dump(us_dict, open(dump_path, 'wb'))
            
        us_dict = defaultdict(lambda :null_dict, us_dict)
        
        return ps_dict, sp_dict, us_dict
    
    
    def get_feature_product_shop(self):
        
        print("正在进行特征查询")
        ps_dict, sp_dict, us_dict = self.get_product_shop()
        
        print("通过sku_id查询商品信息")
        
        self.feats = self.feats.dropna()
        
        for i in range(len(self.fets1) - 1):
            print(i)
            self.feats[self.fets1[i + 1]] = self.feats["sku_id"].map(lambda x:
                    ps_dict[x][self.fets1[i + 1]])
        
        self.feats = self.feats.dropna()
        
        print("通过shop_id查询店铺信息")
        
        for i in range(len(self.fets2) - 1):
            print(i)
            self.feats[self.fets2[i + 1]] = self.feats["shop_id"].map(lambda x:
                    sp_dict[x][self.fets2[i + 1]])
        
        self.feats = self.feats.dropna()
        
        print("通过user_id查询用户信息")
        
        for i in range(len(self.fets3) - 1):
            print(i)
            self.feats[self.fets3[i + 1]] = self.feats["user_id"].map(lambda x:
                    us_dict[x][self.fets3[i + 1]])
           
        self.feats = self.feats.dropna()


def get_f(time):
    dump_path = './qcache/%s_7_30_all.pkl' % time
    test = windows("%s 00:00:00" % time, 7 , 30)
    pickle.dump(test.feats, open(dump_path, 'wb'))
    
if __name__ == "__main__":
    get_f("2018-04-15")
    get_f("2018-04-10")
    get_f("2018-04-06")
    get_f("2018-03-27")
    get_f("2018-03-22")
    get_f("2018-03-15")
    get_f("2018-03-08")
    get_f("2018-03-01")