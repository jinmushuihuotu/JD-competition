# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
import pandas as pd
import numpy as np


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

                     
def get_product_shop(product,shop):
    sku_id = np.ndarray.tolist(np.unique(product.iloc[:,0]))
    shop_id_product = np.ndarray.tolist(np.unique(product.iloc[:,2]))
    shop_id_shop  = np.ndarray.tolist(np.unique(shop.iloc[:,1]))
    i = 0
    j = 0
    sku_id_dict = {}
    while i < len(sku_id):
        sku_id_dict[sku_id[i]] = [product.iloc[i,4]] #存入商品上市时间
        while j < len(shop_id_product):
            if shop_id_product[j] in shop_id_shop == True:
                index0 = shop_id_shop.index(shop_id_product[j])
                sku_id_dict[sku_id[i]] = sku_id_dict[sku_id[i]].append(shop[index0,1], shop[index0,2], shop[index0,2],shop[index0,4],shop[index0,6])
                #增加shop数据中2、3、4、5、7列
    return sku_id_dict

product = pd.read_csv(open('C:\\Users\\leeec\\Documents\\Tencent Files\\978305996\\FileRecv\\jdata\\jdata_product.csv'))
shop = pd.read_csv(open('C:\\Users\\leeec\\Documents\\Tencent Files\\978305996\\FileRecv\\jdata\\jdata_shop.csv'))
comment = pd.read_csv(open('C:\\Users\\leeec\\Documents\\Tencent Files\\978305996\\FileRecv\\jdata\\jdata_comment.csv'))

product.iloc[:,4] =  product.iloc[:,4].apply(time_transform)
shop.iloc[:,4] =  shop.iloc[:,4].apply(time_transform)
comment.iloc[:,0] = comment.iloc[:,0].apply(time_transform0)
     
get_product_shop(product,shop)                

                        