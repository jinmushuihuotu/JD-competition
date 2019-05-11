# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Sat May 11 12:42:05 2019

@author: leeec
"""

# -*- coding: utf-8 -*-
"""
=======
>>>>>>> c9e07b89bf1eb15a12c8f7d58b8e6d491eca302b
Bahn
"""

import time
import numpy as np
import pandas as pd
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
            raise Exception("Error: y and lange must >= 0")
            
        self.y = y
        self.lange = lange
        self.subset = subset
        
        # 提前初始化记录来自原文件的特征信息
        # 商品特征（product文件）
        self.fets1 = ['sku_id', 'cate', 'market_time', 'shop_id']
        
        # 店铺特征（shop文件）
        self.fets2 = ['shop_id', 'cate_s', 'fans_num',
                      'vip_num', "shop_reg_tm", 'shop_score']
        
        # 用户特征（user文件）
        self.fets3 = ["user_id", "user_reg_tm", 'age', 'sex',
                       "user_lv_cd", "city_level",
                       "province", "city", "county"]
        #商品维度新特征
        self.fets5 = ['sku_id','browse_ratio','sale_vol_cate','sale_vol_shop']

        
        # 计算窗口的三个时间点
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
        
        time_end = time.time()
        print('time cost',time_end-time_start,'s')
        
               
    def time_transform(self, times):
        '''
        时间转换函数
        返回：把文本"%Y-%m-%d %H:%M:%S" 转换为时间戳，并减去2018-02-01
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
                self.y,
                self.lange,
                self.subset)
        
        # 查找缓存
        if os.path.exists(dump_path):
            actions_feat = pickle.load(open(dump_path, "rb"))
            return actions_feat

        
        
        print('提取窗口内商品关注时间')
        # 查询关注时间
        tmp1 = self.actions1[['user_id','sku_id','action_time']].groupby(['user_id','sku_id'],
                                                 as_index = False).min()['action_time']
        tmp2 = self.actions1[['user_id','sku_id','action_time']].groupby(['user_id','sku_id'],
                                                 as_index = False).max()['action_time']
        actions_dauer = (tmp2 - tmp1).copy()
        del tmp1
        del tmp2
            
        # 判断是否需要利用第二段行为数据
        if self.y != 0:
            print("查找特征，创建目标变量")
            tp_action = self.actions2[self.actions2["type"] == 2].copy()
            tp_action.loc[:, "type"] = 6
            actions_feat = self.actions1.append(tp_action).copy()
            
        else:
            print("查找特征")
            actions_feat = self.actions1.copy()
            
        # 强制把第一个商品加入购入车，避免没有type=5的样本(数据bug)
        actions_feat.loc[actions_feat.index[0], 'type'] = 5
        
        # 把type转换为虚拟变量
        dums = pd.get_dummies(actions_feat['type'], prefix = 'type')
        actions_feat = pd.concat([actions_feat[['user_id','sku_id']],
                                   dums], axis = 1)
        
        # 累计各行为操作次数
        actions_feat = actions_feat.groupby(['user_id','sku_id'],
                                             as_index = False).sum()
        
        # 清洗只在actions2内购买的产品
        actions_feat = actions_feat[(actions_feat["type_1"] > 0)
                                    | (actions_feat["type_2"] > 0)
                                    | (actions_feat["type_3"] > 0)
                                    | (actions_feat["type_4"] > 0)
                                    | (actions_feat["type_5"] > 0)]
        
        # 插入时间
        actions_dauer.name = 'actions_dauer'
        actions_feat['actions_dauer'] = actions_dauer
        
        # 规范标签值，修改列名
        if self.y != 0:
            actions_feat.loc[actions_feat["type_6"] > 0, "type_6"] = 1
            actions_feat.rename(columns={"type_6": "tar"},
                                inplace=True)
            
            # 删除随便浏览的产品（精度降低，速度显著提升）
            # actions_feat = actions_feat[actions_feat['actions_dauer'] > 0]
            
        
        
        
        pickle.dump(actions_feat, open(dump_path, 'wb'))
        
        return actions_feat
    
    def get_actions(self, start_date, end_date):
        """
        获取时间段内的actions数据
        """
        dump_path = './cache/all_action_%s_%s_%s_%s.pkl' % (start_date,
                                                      end_date,
                                                      self.y,
                                                      self.subset)
        if os.path.exists(dump_path):
            actions = pickle.load(open(dump_path, "rb"))
            return actions
        
        elif os.path.exists('./cache/all_action.pkl'):
            actions = pickle.load(open('./cache/all_action.pkl',
                                     "rb"))#.iloc[0:1000000,:]
        else:
            actions = pd.read_csv(action_path)
            pickle.dump(actions, open('./cache/all_action.pkl', 'wb'))
            #actions = actions.iloc[0:1000000,:]
            
        actions = actions[(actions["action_time"] >= start_date) &
                                  (actions["action_time"] < end_date)]
        pickle.dump(actions, open(dump_path, 'wb'))

        return actions
    
                     
    def get_feature_map(self):
        '''
        获取需要映射查询的特征值
        '''
        null_dict = defaultdict(lambda :np.nan)
        
        # 读取product并转换为字典
        
        dump_path = './cache/basic_product.pkl'
        if os.path.exists(dump_path):
            ps_dict = pickle.load(open(dump_path, "rb"))
            
        else:
            product = pd.read_csv(product_path)            
            product = product[self.fets1]
            ps_dict = product.set_index('sku_id').to_dict('index')
            pickle.dump(ps_dict, open(dump_path, 'wb'))
        
        ps_dict = defaultdict(lambda :null_dict, ps_dict)
        
        # 读取shop并转换为字典
        
        dump_path = './cache/basic_shop.pkl'
        if os.path.exists(dump_path):
            sp_dict = pickle.load(open(dump_path, "rb"))
        else:
            shop = pd.read_csv(shop_path)
            shop = shop[self.fets2]
            sp_dict = shop.set_index('shop_id').to_dict('index')
            pickle.dump(sp_dict, open(dump_path, 'wb'))
            
        sp_dict = defaultdict(lambda :null_dict, sp_dict)
        
        # 读取user并转换为字典
        
        dump_path = './cache/basic_user.pkl'
        if os.path.exists(dump_path):
            us_dict = pickle.load(open(dump_path, "rb"))
        else:
            user = pd.read_csv(user_path)
            user = user[self.fets3]
            us_dict = user.set_index('user_id').to_dict('index')
            pickle.dump(us_dict, open(dump_path, 'wb'))
            
        us_dict = defaultdict(lambda :null_dict, us_dict)
        
        
        # 聚合用户行为特征并转换为字典（动态）
        user_r = self.feats[['user_id', 'type_1', 'type_2',
                             'type_3', 'type_4', 'type_5']].copy()
        user_r = user_r.groupby(['user_id'], as_index = False).sum()
                
        # 不收录没有买过任何东西的用户
        user_r = user_r[user_r['type_2'] > 0]
        
        # 用户转化率
        user_r['user_ratio'] = user_r['type_2'] / (user_r['type_1'] + 
              user_r['type_2'] + user_r['type_3'] + user_r['type_4'] + user_r['type_5'])
        
        ur_dict = user_r.set_index('user_id').to_dict('index')
        ur_dict = defaultdict(lambda :null_dict, ur_dict)
        
        
        # 提取评论特征并转化为字典
        dump_path = './cache/basic_comment.pkl'
        if os.path.exists(dump_path):
            comment = pickle.load(open(dump_path, "rb"))
        else:  
            comment = pd.read_csv(comment_path)
        
        # 获取评论的累计信息
        comment = comment[comment["dt"] < self.mid_date].copy()
        del comment['dt']
        comment = comment.groupby(['sku_id'], as_index = False).sum()
        comment['gut_ratio'] = comment['good_comments'] / comment['comments']
        self.fets4 = list(comment.columns)
        cm_dict = comment.set_index('sku_id').to_dict('index')
        cm_dict = defaultdict(lambda :null_dict, cm_dict)

        return ps_dict, sp_dict, us_dict, ur_dict, cm_dict
    
    
    def get_feature_product_shop(self):
        '''
        通过特征字典查询特征
        注意：np.nan以浮点数储存，会导致数据类型改变，但原字典键值是int，
        这会造成索引速度变慢
        '''
        print("开始进行特征查询")
        # 生成特征查询字典
        ps_dict, sp_dict, us_dict, ur_dict, cm_dict = self.get_feature_map()
        
        if self.y != 0:
            print("清理无购买的用户")
            for i in ['user_ratio']:
                self.feats[i] = self.feats["user_id"].map(lambda x:
                        ur_dict[x][i])
            self.feats.dropna()
        
        print("通过sku_id查询商品信息")
        for i in range(len(self.fets1) - 1):
            print(i)
            self.feats[self.fets1[i + 1]] = self.feats["sku_id"].map(lambda x:
                    ps_dict[x][self.fets1[i + 1]])
        
        # 删掉无法查到shop_id的数据（无法补缺），修正数据类型
        self.feats = self.feats.dropna()
        self.feats.drop_duplicates(inplace=True)
        for i in ['sku_id', 'shop_id', 'cate']:
            self.feats[i] = self.feats[i].astype('int')

        print("通过shop_id查询店铺信息")
        for i in range(len(self.fets2) - 1):
            print(i)
            self.feats[self.fets2[i + 1]] = self.feats["shop_id"].map(lambda x:
                    sp_dict[x][self.fets2[i + 1]])
        
        # 补齐缺失值，修正数据类型
        self.feats.fillna({"shop_reg_tm":self.feats.median()["shop_reg_tm"],
             "shop_score":self.feats.median()["shop_score"],
             "fans_num":self.feats.median()["fans_num"],
             "vip_num":self.feats.median()["vip_num"],
             "cate_s":0}, inplace=True)
    
        for i in ['shop_id', 'fans_num',
          'vip_num', 'cate_s']:
            self.feats[i] = self.feats[i].astype('int')
        
        # 构造新特征，店铺主营品类与商品品类是否相同
        self.feats['is_same'] = self.feats["cate"] - self.feats["cate_s"]
        self.feats.loc[self.feats['is_same'] != 0, 'is_same'] = 1
        
        print("通过user_id查询用户信息")
        for i in range(len(self.fets3) - 1):
            print(i)
            self.feats[self.fets3[i + 1]] = self.feats["user_id"].map(lambda x:
                    us_dict[x][self.fets3[i + 1]])
          
        # 补齐缺失值，修正数据类型
        self.feats.fillna({"user_reg_tm":self.feats.median()["user_reg_tm"]}, inplace = True)
        self.feats.fillna(-999999, inplace=True)
        
        tp = self.fets3.copy()
        del tp[1]
        self.feats.drop_duplicates(inplace=True)
        for i in tp:
            self.feats[i] = self.feats[i].astype('int')
        
        
        print("通过sku_id查找评论数据")
        for i in range(len(self.fets4) - 1):
            print(i)
            self.feats[self.fets4[i + 1]] = self.feats["sku_id"].map(lambda x:
                    cm_dict[x][self.fets4[i + 1]])

                
        print("通过sku_id查询商品维度新特征")
        for i in range(len(self.fets5) - 1):
            print(i)
            self.feats[self.fets5[i + 1]] = self.feats["sku_id"].map(lambda x:
                    sku_level_dict[x][self.fets5[i + 1]])

        
        # 填补缺失值，修正数据类型
        self.feats.fillna({"comments":self.feats.median()["comments"],
             "good_comments":self.feats.median()["good_comments"],
             "bad_comments":self.feats.median()["bad_comments"],
             "gut_ratio":self.feats.median()["gut_ratio"]}, inplace=True)
        
        tp = self.fets4.copy()
        del tp[4]
        self.feats.drop_duplicates(inplace=True)
        for i in tp:
            self.feats[i] = self.feats[i].astype('int')
            
    def get_feature_sku_level(self) :
        #利用self.feats根据商品维度计算新特征，生成sku_level_dict
        
        product_level_data = pd.DataFrame(columns=['sku_id','cate','shop_id','sale_vol','browse_num',
                                                   'browse_ratio','sale_vol_cate','sale_vol_shop'])
        sku_id_dict  = dict(list(self.feats[['sku_id','cate','shop_id','type_1','type_2'
                                             ]].groupby(['sku_id'],as_index = False)))
        
        for key in sku_id_dict:
            product_level_data['sku_id'] = key
            product_level_data['cate'] = sku_id_dict[key]['cate']
            product_level_data['shop_id'] = sku_id_dict[key]['shop_id']
            product_level_data['sale_vol'] = sku_id_dict[key]['type_2'].sum()
            product_level_data['browse_num'] = sku_id_dict[key]['type_1'].sum()
        
        temp1 = product_level_data[['cate','sale_vol']].groupby(['cate'],as_index = False).sum().add_prefix('sum_').reset_index()
        temp2 = product_level_data[['shop_id','sale_vol']].groupby(['shop_id'],as_index = False).sum().add_prefix('sum_').reset_index()
        product_level_data = pd.merge(pd.merge(product_level_data,temp1),temp2)
        del temp1
        del temp2
        
        product_level_data['browse_ratio'] = product_level_data['sale_vol']/product_level_data['browse_num']   
        product_level_data['sale_vol_cate'] = product_level_data['sale_vol']/product_level_data['sum_cate']     
        product_level_data['sale_vol_shop'] = product_level_data['sale_vol']/product_level_data['sum_shop_id'] 
              
        product_level_data = product_level_data[['sku_id','browse_ratio','sale_vol_cate','sale_vol_shop']]  
        sku_level_dict = product_level_data.set_index('sku_id').to_dict(orient='index')






def get_f(time):
    dump_path = './qcache/%s_7_30_all.pkl' % time
    test = windows("%s 00:00:00" % time, 7 , 30)
    pickle.dump(test.feats, open(dump_path, 'wb'))
    
if __name__ == "__main__":
    '''
    get_f("2018-04-10")
    get_f("2018-04-06")
    get_f("2018-03-27")
    get_f("2018-03-22")
    get_f("2018-03-15")
    get_f("2018-03-08")
    get_f("2018-03-01")
    get_f("2018-04-15")
    '''
    test = windows("%s 00:00:00" % "2018-04-15", 7 , 30)