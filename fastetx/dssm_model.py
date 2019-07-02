#coding:utf-8
import pandas as pd

import  keras.preprocessing.sequence as kps
import numpy as np
from keras.layers  import Embedding,LSTM,Input,Dot,Lambda
from keras.models   import Model
from  scipy.stats  import rankdata
import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import  pickle
# 进行配置，每个GPU使用60%上限现存
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
# 设置session
KTF.set_session(session )

def load_pickle(file_name):
        with open(file_name, 'rb') as f:
                return pickle.load(f)

def get_MRR(data):
    GR_COLS = ["user_id", "session_id", "timestamp", "step"]
    def get_rank(line):
        max_rank=line.shape[0]+1.0
        line['rank']=max_rank-rankdata(line.score.values)
        return line
    data=data.groupby(GR_COLS).apply(get_rank)
    rank_=data.query('label==1')
    rank_cnt=rank_['rank'].value_counts()
    mrr=((rank_cnt/rank_cnt.sum())*(1./rank_cnt.index)).sum()
    return mrr


n_steps=20
n_items=262144
max_length=20
vector_size=10


def get_pad(data):
    item_seq=kps.pad_sequences(data.item_seq_enc.values.tolist(),maxlen=20)
    refer_data=kps.pad_sequences(data.impression_id.values.reshape(-1,1).tolist(),maxlen=20)
    label_data=data.label.values.reshape(-1,1)
    return item_seq,refer_data,label_data

def Model():
        item_input = Input(shape=[n_steps], name='item_views')
        impression_input = Input(shape=[n_steps], name="impression_")

        embed = Embedding(input_dim=n_items + 1, output_dim=vector_size,
                          input_length=max_length,mask_zero=True)

        item_embed = embed(item_input)
        item_embed = LSTM(10, activation='relu',
                          input_shape=(max_length, vector_size))(item_embed)

        impression_embed = embed(impression_input)
        impression_embed = Lambda(lambda x: x[:, -1], output_shape=(vector_size,))(impression_embed)
        # impression_embed=Flatten()(impression_embed)
        # impression_embed=GlobalMaxPool1D()(impression_embed)
        prod = Dot(axes=1)([item_embed, impression_embed])

        model = Model([item_input, impression_input], prod)

        model.compile('adam', 'mean_squared_error')
        return model

if __name__=='__main__':

        processed=True
        if not processed:
                data = pd.read_csv('dssm.csv')
                data.item_seq_enc = data.item_seq_enc.map(eval)
                data.impression_id = data.impression_id.astype('int')

                data.item_seq_enc = data.item_seq_enc.map(lambda line: np.array(line) + 1)
                data.impression_id = data.impression_id.map(lambda line: line + 1)
                data_value = data.values
                np.random.shuffle(data_value)
                data = pd.DataFrame(data_value, columns=data.columns)


                train_data = data[data.flag == 'train']
                test_data = data[data.flag == 'test']
        else:
                test_data=pd.read_pickle('./data/test_data.pkl')
                train_data=pd.read_pickle('./data/train_data.pkl')

        item_seq, refer_data, label_data = get_pad(train_data)
        item_seq_valid, refer_data_valid, label_data_valid = get_pad(test_data)


        model=Model()
        history = model.fit([item_seq, refer_data],
                            label_data, shuffle=True,
                            epochs=10,
                            batch_size=100,
                            verbose=1)

        # step valid eval
        y_pre_valid = model.predict([item_seq_valid, refer_data_valid])
        test_data['score'] = y_pre_valid
        test_Mrr = get_MRR(test_data)
        print("Valid MRR :{}".format(test_Mrr))

        y_pre = model.predict([item_seq, refer_data])
        train_data['score'] = y_pre
        train_Mrr = get_MRR(train_data)

        print("Valid MRR :{}".format(test_Mrr))







