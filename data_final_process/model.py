import pandas as pd
import xgboost as xgb





train_path='data_process/train_set.csv'
test_path='data_process/test_data.csv'

train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

Y_valid=test_data.label
X_valid=test_data.drop('label',axis=1)

Y=train_data.label.values
X=train_data.drop('label',axis=1)

xgb_train = xgb.DMatrix(data=X,label=Y)
xgb_val=xgb.DMatrix(data=X_valid,label=Y_valid)




params={'booster':'gbtree',
    'objective': 'rank:pairwise',
    'eval_metric':'auc',
    'gamma':0.1,
    'min_child_weight':1.1,
    'max_depth':7,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'colsample_bylevel':0.7,
    'eta': 0.01,
    'tree_method':'exact',
    'seed':1000,
    'nthread':12
    }

params1={
'booster':'gbtree',
'objective': 'binary:logistic',
'scale_pos_weight': 1/7.5,
#7183正样本
#55596条总样本
#差不多1:7.7这样子
'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':8, # 构建树的深度，越大越容易过拟合
'lambda':3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
#'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.03, # 如同学习率
'seed':1000,
'nthread':12,# cpu 线程数
'eval_metric': 'map'
}


plst = list(params1.items())
num_rounds = 300 # 迭代次数

watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
model = xgb.train(plst, xgb_train,num_boost_round=num_rounds,evals=watchlist,early_stopping_rounds=500,verbose_eval=10)


model.save_model('./models/20180605_xgb.model') # 用于存储训练出的模型
print ("best best_ntree_limit",model.best_ntree_limit)   #did not save the best,why?
print ("best best_iteration",model.best_iteration) #get it?


#
pred=model.predict(xgb_val)
y_pred=(pred>0.5).astype(int)
from sklearn.metrics import classification_report
print(classification_report(Y_valid,y_pred))