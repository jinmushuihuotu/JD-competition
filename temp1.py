# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
import pandas as pd
import numpy as np

comment_path = "data/jdata/jdata_comment.csv"
shop_path = "data/jdata/jdata_shop.csv"
product_path = "data/jdata/jdata_product.csv"
user_path = "data/jdata/jdata_user.csv"

def time_transform(times):
    if len(times) == 21:
        times = times[:-2]     
    ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d %H:%M:%S")) - 1517414400.0
    return ts

#comment时间数据格式不同，定义对应函数
def time_transform0(times):
    if len(times) == 21:
        times = times[:-2]     
    ts = time.mktime(time.strptime('{}'.format(times),
         "%Y-%m-%d")) - 1517414400.0
    return ts

#product对应shop                   
def get_product_shop(product,shop):
    sku_id = np.ndarray.tolist(np.unique(product.iloc[:,0]))
    shop_id_product = np.ndarray.tolist(np.unique(product.iloc[:,2]))
    shop_id_shop  = np.ndarray.tolist(np.unique(shop.iloc[:,1]))
    i = 0
    j = 0
    sku_id_dict = {}
    while i < len(sku_id):
        sku_id_dict[sku_id[i]] = [product.iloc[i,4]] #存入product数据商品上市时间
        while j < len(shop_id_product):
            if shop_id_product[j] in shop_id_shop == True:
                index0 = shop_id_shop.index(shop_id_product[j])
                list0 = [shop.iloc[index0,1], shop.iloc[index0,2], shop.iloc[index0,3],shop.iloc[index0,4],shop.iloc[index0,6]]
                sku_id_dict[sku_id[i]].extend(list0)
                #增加shop数据中2、3、4、5、7列
            j = j + 1
        i = i + 1   
    return sku_id_dict

def get_feature_product_shop(data,dict):
    sku_id =list(data.iloc["sku_id"])
    sku_id0 = list(dict.keys())
    for i in range(len(sku_id )):
        for j in range(len(sku_id0)) :
            if int(sku_id[i]) == int(sku_id0[j]):
                data["market_time"][i] = dict[int(sku_id0[j])][0]
                data["fans_num"][i] = dict[int(sku_id0[j])][2]
                data["vip_num"][i] = dict[int(sku_id0[j])][3]
                data["shop_reg_tm"][i] = dict[int(sku_id0[j])][4]
                data["shop_score"][i] = dict[int(sku_id0[j])][5]
    return data

def get_feature_user(data,user):
    user_id =list(data.iloc["user_id"])
    user_id0 =list(user.iloc["user_id"])
    for i in range(len(user_id)):
        for j in range(len(user_id0)) :
            if int(user_id[i]) == int(user_id0[j]):
                data["age"][i] = user["age"][j]
                data["sex"][i] = user["sex"][j]
                data["user_reg_tm"][i] = user["user_reg_tm"][j]
                data["user_lv_cd"][i] = user["user_lv_cd"][j]
                data["city_level"][i] = user["city_level"][j]
                data["province"][i] = user["province"][j]
                data["city"][i] = user["city"][j]
                data["county"][i] = user["county"][j]
    return data

product = pd.read_csv(product_path)
shop = pd.read_csv(shop_path)
comment = pd.read_csv(comment_path)
user = pd.read_csv(user_path)

product.iloc[:,4] =  product.iloc[:,4].apply(time_transform)
shop.iloc[:,4] =  shop.iloc[:,4].apply(time_transform)
comment.iloc[:,0] = comment.iloc[:,0].apply(time_transform0)
user.iloc[:,5] = user.iloc[:,5].apply(time_transform)
     
dict0 = get_product_shop(product,shop)
newdata = get_feature_product_shop(data,dict0)
newdata = get_feature_user(data,user)