import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


train_path='data_process/train_set.csv'
test_path='data_process/test_data.csv'

train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

Y_valid=test_data.label
X_valid=test_data.drop('label',axis=1)

Y=train_data.label.values
X=train_data.drop('label',axis=1)



train_data = lightgbm.Dataset(X, label=Y,categorical_feature=['hour'])
test_data = lightgbm.Dataset(X_valid, label=Y_valid)


parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


model.save_model('models/lgb500.pkl')
#



