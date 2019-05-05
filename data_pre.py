import pandas as pd
from collections import Counter
import pickle

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
数据排序部分
data = pd.read_csv(action_path)
data.iloc[:, 2] =  data.iloc[:, 2].map(time_transform)
data = data.sort_values(by = ["user_id",
                                "action_time"])
data.to_csv("jdata_action.csv", index = False)

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