from keras.layers import Dense, Embedding, LSTM, merge, Input, GlobalMaxPool1D, Dot, Flatten, Lambda, Multiply
from keras.models import Sequential, Model
import keras.preprocessing.sequence   as kps
import pandas as pd
from scipy.stats import rankdata
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import argparse
import pickle
from scipy import sparse
import time
from keras.utils.training_utils import multi_gpu_model

n_items = 475285
max_length = 50
vector_size = 10
n_action = 6


def load_pickle(file_name):
        with open(file_name, 'rb') as f:
                return pickle.load(f)


def to_pickle(obj, file_name):
        with open(file_name, 'wb') as f:
                pickle.dump(obj, f)


def transform(data):
        data.item_list = data.item_list.map(eval)
        data.action_list = data.action_list.map(eval)
        item_list = kps.pad_sequences(data.item_list.values, maxlen=max_length)
        impression2id = kps.pad_sequences(data.impression2id.values.reshape(-1, 1), maxlen=max_length)
        action_list = kps.pad_sequences(data.action_list.values, maxlen=max_length)
        label_data = data.label.values.reshape(-1, 1)
        return [item_list, impression2id, action_list], label_data


def weight_dssm():
        item_input = Input(shape=[max_length], name='item_views')
        action_input = Input(shape=[max_length])

        impression_input = Input(shape=[max_length], name="impression_")

        action_embedding = Embedding(input_dim=n_action + 1, output_dim=1,
                                     input_length=max_length, mask_zero=True, name='action_embed')

        item_embedding = Embedding(input_dim=n_items + 1, output_dim=vector_size,
                                   input_length=max_length, mask_zero=True, name='item_embed')

        action_embed = action_embedding(action_input)

        item_embed = item_embedding(item_input)

        item_embed = Multiply()([action_embed, item_embed])

        item_embed = LSTM(vector_size, activation='relu',
                          input_shape=(max_length, vector_size))(item_embed)

        impression_embed = item_embedding(impression_input)
        impression_embed = Lambda(lambda x: x[:, -1], output_shape=(vector_size,))(impression_embed)
        # # impression_embed=Flatten()(impression_embed)
        # # impression_embed=GlobalMaxPool1D()(impression_embed)
        prod = Dot(axes=1)([item_embed, impression_embed])
        prod = Dense(1, activation='sigmoid', name='prediction')(prod)

        model = Model([item_input, impression_input, action_input], prod)

        return model


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


def eval_transform(data):
        item_list = kps.pad_sequences(data.item_list.values, maxlen=max_length)
        impression2id = kps.pad_sequences(data.impression2id.values.reshape(-1, 1), maxlen=max_length)
        action_list = kps.pad_sequences(data.action_list.values, maxlen=max_length)
        label_data = data.label.values.reshape(-1, 1)
        return [item_list, impression2id, action_list], label_data


def get_submission(sub_data):
        GR_COLS = ["user_id", "session_id", "timestamp", "step"]

        def group_concat(df, gr_cols, col_concat):
                """Concatenate multiple rows into one."""

                df_out = (
                        df
                                .groupby(gr_cols)[col_concat]
                                .apply(lambda x: ' '.join(x))
                                .to_frame()
                                .reset_index()
                )

                return df_out

        df_out = (
                sub_data
                        .assign(impressions=lambda x: x["impression"].apply(str))
                        .sort_values(GR_COLS + ["score"],
                                     ascending=[True, True, True, True, False])
        )

        df_out = group_concat(df_out, GR_COLS, "impressions")
        df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)
        return df_out


if __name__ == '__main__':

        parse = argparse.ArgumentParser()
        parse.add_argument("--data_path", type=str, default="./data")
        parse.add_argument('--epochs', type=int, default=100)
        parse.add_argument('--batch_size', type=int, default=500)
        parse.add_argument('--model_pth', type=str, default='./checkpoint')
        parse.add_argument('--use_gpu', type=int, default=1)
        parse.add_argument('--finetune', type=int, default=0)
        parse.add_argument('--train_pickle', type=str, default='train_data.pkl')
        parse.add_argument('--gpu_index', type=str, default='3')
        args = parse.parse_args()
        data_pth = args.data_path
        time_local = time.localtime(time.time())
        dt = time.strftime("%Y-%m-%d-%H:%M", time_local)

        # 进行配置，每个GPU使用60%上限现存
        if args.use_gpu == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
                config = tf.ConfigProto()
                config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 每个GPU现存上届控制在60%以内
                session = tf.Session(config=config)
                # 设置session
                KTF.set_session(session)

        if len(args.train_pickle) < 3:
                data = pd.read_csv(os.path.join(data_pth, "dssm_train.csv"))
                X, Y = transform(data)
                test_data = pd.read_csv(os.path.join(data_pth, 'dssm_test.csv'))
                test_X, test_Y = transform(test_data)

        else:
                X, Y, test_X, test_Y = load_pickle(os.path.join(data_pth, args.train_pickle))
                X = [line.toarray() for line in X]
                test_X = [line.toarray() for line in test_X]

        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        checkpoint = ModelCheckpoint(filepath='checkpoint/dssm.h5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4,
                                           mode='min')
        tb = TensorBoard(log_dir='./checkpoint')
        callback_lists = [checkpoint, tb, earlyStopping, reduce_lr_loss]

        # step 1 train model
        print('**--**')
        print("start train.....")
        model = weight_dssm()
        if args.finetune == 1:
                model.load_weights('checkpoint/dssm.h5')

        model.compile('adam', 'mean_squared_error', metrics=['accuracy'])
        params = {'epochs': args.epochs,
                  "batch_size": args.batch_size,
                  'shuffle': True, 'verbose': 1,
                  'callbacks': callback_lists, 'validation_data': [test_X, test_Y]}

        model.fit(X, Y, **params)
        model.save(os.path.join(args.model_pth, 'dssm_%s.h5' % dt))

        # step2 :evalate
        print("evalate start......")
        valid_data = pd.read_csv(os.path.join(data_pth, 'dssm_valid.csv'))
        valid_X, valid_Y = transform(valid_data)
        valid_data['score'] = model.predict(valid_X)
        mrr = get_MRR(valid_data)
        print("valid MRR:", mrr)

        # submissions
        sub_miss = pd.read_pickle("../data/dssm_submission.pkl")
        # sub_miss1=sub_miss.head(1000)
        sub_X, sub_Y = eval_transform(sub_miss)
        sub_miss['score'] = model.predict(sub_X)
        sub_ = get_submission(sub_miss)

        sub_.to_csv("../data/sub_%s" % dt, index=False)
