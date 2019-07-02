# coding:utf-8
import tensorflow as tf
from keras.layers import Dense, Embedding
import pandas as pd
import os
from keras.layers import GlobalAvgPool1D, Lambda, Dot, Dense, Concatenate, Input
from keras import backend as K
from keras import Model
import keras.preprocessing.sequence as kps
from scipy.stats import rankdata
import argparse
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping,ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import pickle

max_length = 25
n_items = 475285
vector_size = 10


def load_pickle(file_name):
        with open(file_name, 'rb') as f:
                return pickle.load(f)


def to_pickle(obj, file_name):
        with open(file_name, 'wb') as f:
                pickle.dump(obj, f)


def loc_model(fine_weights):
        impressions_input = Input(shape=[max_length], name="impressions_input")
        loc_input = Input(shape=[1], name='loc_input')
        impression_input = Input(shape=[1], name="impression_input")
        # input_length 不设置可以接受任意长度
        item_embedding = Embedding(input_dim=n_items + 1, output_dim=vector_size, weights=[fine_weights]
                                   , mask_zero=True, name='item_embedding',trainable=True)

        item_embed = item_embedding(impressions_input)

        avg_embed = GlobalAvgPool1D(name='avg-embed')(item_embed)
        avg_embed = Dense(vector_size, activation='relu')(avg_embed)

        impression_embed = item_embedding(impression_input)
        impression_embed = Lambda(lambda r: K.squeeze(r, 1))(impression_embed)

        prediction = Dot(axes=1)([impression_embed, avg_embed])
        prediction = Concatenate()([prediction, loc_input])
        prediction = Dense(1, activation='sigmoid', name='prediction')(prediction)

        model = Model([impressions_input, loc_input, impression_input], prediction)
        return model


def loc_transform(data):
        impressions = kps.pad_sequences(data.impressions.values, maxlen=max_length)
        impression = data.impression2id.values.reshape(-1, 1)
        locations = kps.pad_sequences(data.impress_rank.values.reshape(-1, 1))
        label_data = data.label.values.reshape(-1, 1)
        return [impressions, locations, impression], label_data


def get_MRR(data):
        GR_COLS = ["user_id", "session_id", "timestamp", "step"]

        def get_rank(line):
                max_rank = line.shape[0] + 1.0
                line['rank'] = max_rank - rankdata(line.score.values)
                return line

        data = data.groupby(GR_COLS).apply(get_rank)
        rank_ = data.query('label==1')
        rank_cnt = rank_['rank'].value_counts()
        mrr = ((rank_cnt / rank_cnt.sum()) * (1. / rank_cnt.index)).sum()
        return mrr


def parse_args():
        parse = argparse.ArgumentParser()
        parse.add_argument("--data_path", type=str, default="./data")
        parse.add_argument('--epochs', type=int, default=100)
        parse.add_argument('--batch_size', type=int, default=500)
        parse.add_argument('--model_pth', type=str, default='./checkpoint')
        parse.add_argument('--use_gpu', type=int, default=1)
        parse.add_argument('--finetune', type=int, default=0)
        parse.add_argument('--model_name', type=str, default='loc_impress')
        args = parse.parse_args()
        return args


def get_callback(model_name):
        checkpoint = ModelCheckpoint(filepath='checkpoint/{}.h5'.format(model_name), monitor='loss', mode='auto',
                                     period=5)
        tb = TensorBoard(log_dir='./checkpoint')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001)

        return [checkpoint, tb, es,reduce_lr]


if __name__ == '__main__':
        # python loc_impress.py --data_path ./nocontext_model/data/demo/ --epochs 1 --batch_size 100 --use_gpu 0
        args = parse_args()
        data_path = args.data_path

        join_path = lambda pth: os.path.join(data_path, pth)
        data = pd.read_parquet(join_path('impress_train'))
        valid_data = pd.read_parquet(join_path('impress_valid'))
        test_data = pd.read_parquet(join_path('impress_test'))
        X, Y = loc_transform(data)
        valid_X, valid_Y = loc_transform(valid_data)
        test_X, test_Y = loc_transform(test_data)  # 94784

        print("Train data size:{} Test data Size :{}  valid data Size :{}".format(data.shape[0], test_data.shape[0],
                                                                                  valid_data.shape[0]))
        if args.use_gpu == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 每个GPU现存上届控制在60%以内
                session = tf.Session(config=config)
                # 设置session
                KTF.set_session(session)

        model_name = "loc_impress_{}_{}".format(args.epochs, args.batch_size)
        callback_lists = get_callback(model_name=model_name)
        # finefune
        weights = load_pickle(join_path('dssm_10_500.pkl'))
        model = loc_model(fine_weights=weights)

        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        params = {'epochs': args.epochs,
                  "batch_size": args.batch_size,
                  'shuffle': True, 'verbose': 1,
                  'callbacks': callback_lists,
                  'validation_data': [test_X, test_Y]
                  }

        model.fit(X, Y, **params)

        valid_data['score'] = model.predict(valid_X)
        mrr = get_MRR(valid_data)

        print("Mean Reciprocal Rank  {}".format(mrr))
