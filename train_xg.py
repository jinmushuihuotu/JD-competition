import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,train_test_split
import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#训练xgb
def xgb_n_train(filename,tar_name='tar'):
    
    data=pickle.load(open(filename,'rb'))
    X=data.drop(columns=tar_name).copy()
    y=data[tar_name].copy()
    y[y>1]=1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 500
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = list(param.items())
        
    plst +=[('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
        
    pickle.dump(bst, open('bst_xgb_%s'%filename[-19:-12], 'wb'))
    

'''
假装是一个帮助文档：
step1:找到数据名称
step2:输入函数
step3:把多出来的bst_xgb_xxxxx.pkl文件发群里
step4:我们完成了一波伪并行
'''
