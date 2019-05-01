import pandas as pd
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.linear_model.logistic import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#读取、整合六个xgb分类器在LR分类器训练数据上得到的结果
def traindata_LR(num=6):
    preds=pd.DataFrame()
    for i in range(num):
        dump_path = './twtrain_result/data%s_results.pkl' % str(i+1)
        pred_LR = pickle.load(open(dump_path, 'rb'))
        preds[i]=pred_LR[0]
    label_LR=pred_LR[['user_id','sku_id']]
    target_LR=pred_LR['tar']
    print(preds)
    pickle.dump(preds, open('./LRtrain/preds.pkl', 'wb'))
    pickle.dump(label_LR, open('./LRtrain/label_LR.pkl', 'wb'))
    pickle.dump(target_LR, open('./LRtrain/tar_LR.pkl', 'wb'))
    return preds,label_LR,target_LR

#根据整合过后的六个概率和LR训练数据集的true_label训练LR分类器并缓存LR分类器
def train_LR():
    preds=pickle.load(open('./LRtrain/preds.pkl', 'rb'))
    target_LR=pickle.load(open('./LRtrain/tar_LR.pkl', 'rb'))
    penaltys = ['l1', 'l2']
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    tuned_parameters = dict(penalty=penaltys, C=Cs)
    
    lr_penalty = LogisticRegression()
    grid = GridSearchCV(lr_penalty, tuned_parameters, cv=5, scoring='f1')
    
    grid.fit(preds, target_LR)
    # print(grid.best_params_)
    # print(grid.best_estimator_.coef_)
    pickle.dump(grid.best_estimator_, open('./LRtrain/best_esti_LR.pkl', 'wb'))
    return grid.best_estimator_

#输入预测活跃用户人数under_num，从六个xgb到LR预测一圈，得出结果，保存到csv文件
def LR_result(under_num,num=6):
    X_test=pd.read_csv('./data_test.csv')
    X_test_label=X_test[['user_id','sku_id']].copy()
    X_test=X_test.drop(columns=['user_id','sku_id']).copy()
    
    X_test_LRdatas=pd.DataFrame(index=X_test_label.index)
    for i in range(num):
        estimator_xgb=pickle.load(open('./twtrain_result/data%s_best_estim.pkl'%(i+1), 'rb'))
        X_test_LRdata=pd.DataFrame(estimator_xgb.predict_proba(X_test)[:,1],columns=[str(i+1)])
        X_test_LRdatas=X_test_LRdatas.join(X_test_LRdata)
        
        
    estimator_LR = pickle.load(open('./LRtrain/best_esti_LR.pkl', 'rb'))
    final_result_pred=estimator_LR.predict_proba(X_test_LRdatas)[:,1]
    final_result_pred=pd.DataFrame(final_result_pred).join(X_test_label)
    
    #print(final_result_pred)
    final_result_pred=final_result_pred.sort_values(by=[0],ascending=False)
    final_results=final_result_pred.iloc[0:under_num,:]
    final_results.to_csv('final_results.csv')
    return final_results

'''
小实测部分
'''
traindata_LR()
train_LR()
LR_result(40)