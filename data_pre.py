import pandas as pd
import numpy as np
from collections import Counter
import featuretools as ft
import pickle
import time


action_path = "data/jdata/jdata_action.csv"
comment_path = "data/jdata/jdata_comment.csv"
product_path = "data/jdata/jdata_product.csv"
user_path = "data/jdata/jdata_user.csv"
shop_path = "data/jdata/jdata_shop.csv"

# 针对shop

def time_transform(times):
    if times is np.nan:
        return np.nan
    
    if len(times) == 21:
        times = times[:-2]
            
    ts = time.mktime(time.strptime('{}'.format(times),
        "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01

    return ts


product = pd.read_csv(product_path, parse_dates = ['market_time'])
product["market_time"] = product["market_time"].map(time_transform)
#product.fillna(-999999, inplace=True)
product.drop_duplicates(inplace=True)
for i in ['sku_id', 'brand', 'shop_id', 'cate']:
    product[i] = product[i].astype('int')
product.to_csv(product_path, index=False)



shop = pd.read_csv(shop_path, na_values=[-1,0])
shop["shop_reg_tm"] = shop["shop_reg_tm"].map(time_transform)
shop.dropna(inplace=True)
'''
shop.fillna({"shop_reg_tm":shop.median()["shop_reg_tm"],
             "shop_score":shop.median()["shop_score"],
             "fans_num":shop.median()["fans_num"],
             "vip_num":shop.median()["vip_num"],
             "cate":0}, inplace=True)
'''
shop.drop_duplicates(inplace=True)

shop.rename(columns={"cate": "cate_s"},
                            inplace=True)

for i in ['vender_id', 'shop_id', 'fans_num',
          'vip_num', 'cate_s']:
    shop[i] = shop[i].astype('int')
shop.to_csv(shop_path, index=False)

user = pd.read_csv(user_path)
user["user_reg_tm"] = user["user_reg_tm"].map(time_transform)
user.fillna({"user_reg_tm":user.median()["user_reg_tm"]}, inplace = True)
user.fillna(-999999, inplace=True)
user.drop_duplicates(inplace=True)
for i in ['user_id', 'age', 'sex', 'user_lv_cd', 'city_level',
       'province', 'city', 'county']:
    user[i] = user[i].astype('int')
user.to_csv(user_path, index=False)



comment = pd.read_csv(comment_path)
def time_transform(times):
    if times is np.nan:
        return np.nan
    
    if len(times) == 21:
        times = times[:-2]
            
    ts = time.mktime(time.strptime('{}'.format(times + " 00:00:00"),
        "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01

    return ts
comment["dt"] = comment["dt"].map(time_transform)
comment.to_csv(comment_path, index=False)







# Create new entityset
es = ft.EntitySet(id = 'actions')

# Create an entity from the client dataframe
# This dataframe already has an index and a time index
'''
es = es.entity_from_dataframe(entity_id = 'actions', 
                              dataframe = actions,
                              variable_types = {'type': ft.variable_types.Categorical},
                              index = 'user_id',
                              time_index = 'action_time')
'''

es = es.entity_from_dataframe(entity_id = 'product', dataframe = product, 
                              index = 'shop_id', time_index = 'market_time')

es = es.entity_from_dataframe(entity_id = 'shop', dataframe = shop, 
                              index = 'shop_id', time_index = 'shop_reg_tm')

r_client_previous = ft.Relationship(es['product']['shop_id'],
                                    es['shop']['shop_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_client_previous)

'''
data = pd.read_csv(action_path)
data.iloc[:, 2] =  data.iloc[:, 2].map(time_transform)
data = data.sort_values(by = ["user_id",
                                "action_time"])
data.to_csv("jdata_action.csv", index = False)


df[df.isnull().values==True].drop_duplicates()
def time_transform(self, times):

    if len(times) == 21:
        times = times[:-2]
            
    ts = time.mktime(time.strptime('{}'.format(times),
        "%Y-%m-%d %H:%M:%S")) - 1517414400.0 #2018-02-01

    return ts
    


'''

'''
action_path = "data/jdata/jdata_action.csv"
data = pd.read_csv(action_path)
def get_from_action_data(df):
    def add_type_count(group):
        behavior_type = group.type.astype(int)
        # 用户行为类别
        type_cnt = Counter(behavior_type)
        # 1: 浏览 2: 下单 3: 关注
        # 4: 评论 5: 加购
        group['browse_num'] = type_cnt[1]
        group['buy_num'] = type_cnt[2]
        group['follow_num'] = type_cnt[3]
        group['comment_num'] = type_cnt[4]
        group['addcart_num'] = type_cnt[5]
        
        return group[['user_id', 'buy_num', "browse_num",
                      'follow_num', 'comment_num', 'addcart_num']]
        
    df_ac = df.groupby(['user_id'],
                        as_index = False).apply(add_type_count)
    # 将重复的行丢弃
    df_ac = df_ac.drop_duplicates('user_id')
    
    return df_ac

data = get_from_action_data(data)
data = data[data['buy_num'] != 0]
pickle.dump(data, open('./cache/all_action.pkl', 'wb'))
'''


'''
数据排序部分


用于绘制浏览-下单时间直方图
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


'''

    def get_from_action_data(self, df):

        def add_type_count(group):
            behavior_type = group.type.astype(int)
            # 用户行为类别
            type_cnt = Counter(behavior_type)
            # 1: 浏览 2: 下单 3: 关注
            # 4: 评论 5: 加购
            group['browse_num'] = type_cnt[1]
            group['buy_num'] = type_cnt[2]
            group['follow_num'] = type_cnt[3]
            group['comment_num'] = type_cnt[4]
            group['addcart_num'] = type_cnt[5]
        
            return group[['user_id', "sku_id", 'buy_num',
                          "browse_num", 'follow_num',
                          'comment_num', 'addcart_num']]
        # df_ac = df.groupby(['user_id',"sku_id"],
        #         as_index = False).agg({"type":Counter})
        df_ac = df.groupby(['user_id',"sku_id"],
                        as_index = False).apply(add_type_count)
        # 将重复的行丢弃
        df_ac = df_ac.drop_duplicates(['user_id','sku_id'])
    
        return df_ac
'''


'''
# 查询当前遍历用户
        nun_user = self.actions1.loc[self.actions1.index[0],
                                "user_id"]
        
        temp1 = {} # 初次浏览时间
        temp2 = {} # 末次浏览时间
        temp3 = defaultdict(lambda: 0) # 总浏览次数
        view_dauer = [] # 浏览时长
        view_times = [] # 浏览次数
        user_id = [] # 用户ID
        sku_ids = [] # 商品ID
    
        for i in self.actions1.index:
            # 如果当前遍历用户改变，结算浏览时长，更新缓存数据
            if self.actions1.loc[i, "user_id"] != nun_user:
                skus = list(temp1.keys())
                if len(skus) <= 1:
                    nun_user = self.actions1.loc[i, "user_id"]
                    temp1 = {}
                    temp2 = {}
                    temp3 = defaultdict(lambda: 0)
                    continue
                for sku in skus:
                    dauer = temp2[sku] - temp1[sku]
                    dauer = int(dauer/60) + 1    
                    view_dauer.append(dauer)
                    view_times.append(temp3[sku])
                    sku_ids.append(sku)
                    user_id.append(nun_user)   
                nun_user = self.actions1.loc[i, "user_id"]
                temp1 = {}
                temp2 = {}
                temp3 = defaultdict(lambda: 0)
                
            sku_id = self.actions1.loc[i, "sku_id"]
            temp3[sku_id] += 1
            temp2[sku_id] = self.actions1.loc[i, "action_time"]
            if sku_id not in temp1.keys():
                temp1[sku_id] = self.actions1.loc[i, "action_time"]
'''

'''
        if self.__y == 0:
            pickle.dump(actions_feat, open(dump_path, 'wb'))
            return actions_feat
        
        
        a2_tar = self.get_from_action_data(self.actions2)
        def tarit(x):
            where_buy = a2_tar[(a2_tar["user_id"] == x['user_id']) 
            & (a2_tar["sku_id"] == x['sku_id'])]["buy_num"]
            if (len(where_buy) > 0) and (list(where_buy)[0] > 0):
                return 1
            else:
                return 0
            
        actions_feat['tar'] = actions_feat.apply(tarit, axis=1)
        
        tar = []
        for i in actions_feat.index:
            has_2 = np.array(self.actions2[(self.actions2["user_id"] == 
                             actions_feat.loc[i, "user_id"]) &
                             (self.actions2["sku_id"] == 
                             actions_feat.loc[i, "sku_id"])]["type"])
            if 2 in has_2:
                tar.append(1)
            else:
                tar.append(0)
                
        actions_feat.insert(0, "tar", tar)
        '''