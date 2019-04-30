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

action_path = "data/jdata/jdata_action.csv"
comment_path = "data/jdata/jdata_comment.csv"
product_path = "data/jdata/jdata_product.csv"
user_path = "data/jdata/jdata_user.csv"
shop_path = "data/jdata/jdata_shop.csv"




class windows():
    
    def __init__(self, end_date,
                     y = 5, lange = 30,
                     subset = 1000000):
        '''
        y = 取目标变量的间隔（天）
        lange = 窗口长度（天）
        start_date：窗口开始时间
        mid_date：窗口中段时间
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
        self.fets3 = ["user_id", "age", "sex", "user_reg_tm",
                       "user_lv_cd", "city_level", "province",
                       "province", "city", "county"]
        
        self.end_date = self.time_transform(end_date)
        self.mid_date = self.end_date - 86400 * y
        self.start_date = self.mid_date - 86400 * lange
        print("寻找行为数据")
        self.actions1 = self.get_actions(self.start_date,
                                        self.mid_date)
        self.actions2 = self.get_actions(self.mid_date,
                                        self.end_date)
        self.feats = self.get_action_feat()
        self.get_feature_product_shop()
        if y != 0:
            self.lower_sample_data()
        
        time_end = time.time()
        print('time cost',time_end-time_start,'s')
        
    def lower_sample_data(self, percent=1):
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
        
        print("遍历行为数据，查找特征")
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
        
        if self.__y == 0:
            pickle.dump(actions_feat, open(dump_path, 'wb'))
            return actions_feat
        
        print("创建目标变量")  
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
        dump_path = './cache/all_action_%s_%s_%s_%s.pkl' % (start_date,
                                                      end_date,
                                                      self.__y,
                                                      self.subset)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, "rb"))
        elif os.path.exists('./cache/all_action.pkl'):
            actions = pickle.load(open('./cache/all_action.pkl',
                                     "rb")).iloc[0:self.subset,:]
        else:
            actions = pd.read_csv(action_path).iloc[0:self.subset,:]
            
        actions = actions[(actions["action_time"] >= start_date) &
                                  (actions["action_time"] < end_date)]
        pickle.dump(actions, open(dump_path, 'wb'))

        return actions
    
                     
    def get_product_shop(self):
        
       
        print("通过sku_id查询商品信息")
        dump_path = './cache/basic_product.pkl'
        if os.path.exists(dump_path):
            ps_dict = pickle.load(open(dump_path, "rb"))
        else:
            product = pd.read_csv(product_path)
            product = product.dropna()
            product["market_time"] = product["market_time"].map(
                    lambda x: self.time_transform(x))
            
            product = product[self.fets1]
            ps_dict = product.set_index('sku_id').T.to_dict('list')
            pickle.dump(ps_dict, open(dump_path, 'wb'))
            
        ps_dict = defaultdict(lambda :np.full([len(self.fets1) - 1],
                                               np.nan), ps_dict)
        
        print("通过shop_id查询店铺信息")
        dump_path = './cache/basic_shop.pkl'
        if os.path.exists(dump_path):
            sp_dict = pickle.load(open(dump_path, "rb"))
        else:
            shop = pd.read_csv(shop_path)
            shop = shop.dropna()
            shop["shop_reg_tm"] = shop["shop_reg_tm"].map(
                    lambda x: self.time_transform(x))
            shop = shop[self.fets2]
            sp_dict = shop.set_index('shop_id').T.to_dict('list')
            pickle.dump(sp_dict, open(dump_path, 'wb'))
            
        sp_dict = defaultdict(lambda :np.full([len(self.fets2) - 1],
                                               np.nan), sp_dict)
        
        
        print("通过user_id查询用户信息")
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
            us_dict = user.set_index('user_id').T.to_dict('list')
            pickle.dump(us_dict, open(dump_path, 'wb'))
            
        us_dict = defaultdict(lambda :np.full([len(self.fets3) - 1],
                                               np.nan), us_dict)    
        return ps_dict, sp_dict, us_dict
    
    
    def get_feature_product_shop(self):
        
        print("正在进行特征查询")
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
           
        self.feats = self.feats.dropna()


if __name__ == "__main__":
    # 
    test = windows("2018-04-15 00:00:00", 5 , 25)
    #test.feats.to_csv("expm.csv", index = False)

