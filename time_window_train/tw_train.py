import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint,uniform
import pickle
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#输入csv文件名，读出x y label
def read_pre_data(filename):
    dump_path_x = './twtrain_data/%s_x.pkl' % filename.split('.')[0]
    dump_path_y = './twtrain_data/%s_y.pkl' % filename.split('.')[0]
    dump_path_label = './twtrain_data/%s_label.pkl' % filename.split('.')[0]
    if os.path.exists(dump_path_x) & os.path.exists(dump_path_y)& os.path.exists(dump_path_label):
         X= pickle.load(open(dump_path_x,'rb'))
         y = pickle.load(open(dump_path_y,'rb'))
         label=pickle.load(open(dump_path_label,'rb'))
    else:
        data=pd.read_csv(filename)
        y=data['tar'].copy()
        X=data.drop(columns=['tar','user_id','sku_id']).copy()
        label=data[['tar','user_id','sku_id']].copy()

        pickle.dump(X, open(dump_path_x, 'wb'))
        pickle.dump(y, open(dump_path_y, 'wb'))
        pickle.dump(label, open(dump_path_label, 'wb'))
    return X,y,label

#根据前六个时间窗口（的某一个，分别）训练六个xgb，然后predict用于训练LR的数据，得到其六个xgb模型的概率
#返回在LR训练数据集上的概率、训练出的xgb分类器，以及分类器的feature_importance
def pred_result_n(X,y,params_dist_grid,params_fixed):
    rs_grid = RandomizedSearchCV(
        estimator=XGBClassifier(**params_fixed, seed=0),
        param_distributions=params_dist_grid,
        n_iter=10,
        cv=5,
        scoring='f1',
        random_state=0
    )
    X_LR, y_LR, label_LR = read_pre_data('data_LR.csv')
    
    rs_grid.fit(X,y)
    #print(rs_grid.best_params_)
    feat_imp = pd.DataFrame(rs_grid.best_estimator_.feature_importances_,index=X.columns)
    feat_imp=feat_imp.sort_values(by=[0],ascending=False)
    
    xgb_pred_n = rs_grid.predict_proba(X_LR)
    xgb_pred_n=pd.DataFrame(xgb_pred_n[:,1]).join(label_LR)
    return xgb_pred_n,rs_grid.best_estimator_,feat_imp


#整合函数，输入xgb训练集、随机网格搜索的参数，保存下来xgb训练器、训练器在LR上的预测结果、feature_importance
def results_to_csv(filename_pre,params_fixed,params_dist_grid):
    
    X,y,label=read_pre_data(filename_pre)
    resultsn,best_est,feat_imp=pred_result_n(X,y,params_dist_grid,params_fixed)

    pickle.dump(resultsn, open('./twtrain_result/%s_results.pkl' % filename_pre.split('.')[0], 'wb'))
    pickle.dump(best_est, open('./twtrain_result/%s_best_estim.pkl' % filename_pre.split('.')[0], 'wb'))
    feat_imp.to_csv('./twtrain_result/%s_feat_imp.csv'% filename_pre.split('.')[0])


'''
小实测部分
'''
params_fixed = {
        'objective': 'binary:logistic',
        'silent': 1
    }
params_dist_grid = {
        'max_depth': [1, 2, 3, 4],
        'gamma': [0, 0.5, 1],
        'n_estimators': randint(1, 1001),
        'learning_rate': uniform(),
        'subsample': uniform(),
        'colsample_bytree': uniform()
        }

results_to_csv('data1.csv',params_fixed,params_dist_grid)

