import tensorflow as tf
import keras
import os
from utils import *
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import *
import tqdm, math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from copy import copy

from keras.layers  import Embedding
class GRU4REC(object):
        def __init__(self, sess,
                     impression_length=25,
                     max_to_keep=5,
                     args=None):

                self.sess = sess
                self.item_series_length = args.item_series_length
                self.embedding_size = args.embedding_size
                self.num_hidden = args.num_hidden
                self.dropout_p_hidden = args.dropout_p_hidden
                self.max_to_keep = max_to_keep
                self.impression_length = impression_length
                self.n_action = args.n_action
                self.n_items = args.n_items
                self.checkpoint_dir = args.checkpoint_dir
                self.data_size = args.data_size
                if not os.path.isdir(self.checkpoint_dir):
                        raise Exception("[!] Checkpoint Dir not found")
                self.batch_size = args.batch_size
                self.build_model()
                self.sess.run(tf.global_variables_initializer())

                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        def build_model(self):

                with tf.name_scope("Input"):
                        self.Input_Item = tf.placeholder(dtype=tf.int32, shape=[None, self.item_series_length],
                                                         name='item_input')
                        self.Input_action = tf.placeholder(dtype=tf.int32, shape=[None, self.item_series_length],
                                                           name='action_input')
                        self.Input_Impression = tf.placeholder(dtype=tf.int32, shape=[None, self.impression_length],
                                                               name='impression_input')
                        self.Input_label = tf.placeholder(tf.int32, shape=[None, self.impression_length],
                                                          name="label_input")

                with tf.name_scope("Embedding"):
                        # initializer = tf.random_normal_initializer(mean=0, stddev=1.0)
                        initializer = tf.glorot_normal_initializer()
                        embedding_item = tf.get_variable('embedding_item', [self.n_items, self.embedding_size],
                                                         initializer=initializer)
                        embedding_action = tf.get_variable('embedding_action', [self.n_action, 1],
                                                           initializer=initializer)

                        softmax_W = tf.get_variable('softmax_w', [self.n_items, self.num_hidden],
                                                    initializer=initializer)

                with tf.name_scope("MutiplyBatchNorm"):
                        Item_embeded = tf.nn.embedding_lookup(embedding_item, self.Input_Item)
                        action_embeded = tf.nn.embedding_lookup(embedding_action, self.Input_action)

                        Item_embeded = tf.multiply(Item_embeded, action_embeded)
                        # bathc normalization
                        Item_embeded = keras.layers.BatchNormalization()(Item_embeded)

                with tf.name_scope("dynamicGRU"):
                        cell = tf.nn.rnn_cell.GRUCell(self.num_hidden, activation=tf.nn.relu)
                        drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
                        len_seq = length(tf.expand_dims(self.Input_Item, 2))
                        init_state = cell.zero_state(self.batch_size, tf.float32)
                        output, state = tf.nn.dynamic_rnn(
                                drop_cell,
                                Item_embeded,
                                initial_state=init_state,
                                dtype=tf.float32,
                                sequence_length=len_seq,
                        )
                        with tf.name_scope("LastOutput"):
                                # 获取最后的输出
                                batch_range = tf.range(tf.shape(output)[0])
                                indices = tf.stack([batch_range, len_seq - 1], axis=1)
                                last = tf.gather_nd(output, indices)

                with tf.name_scope("NegativeSampling"):
                        sampled_W = tf.nn.embedding_lookup(softmax_W, self.Input_Impression)
                        logits = tf.squeeze(tf.matmul(tf.expand_dims(last, 1), tf.transpose(sampled_W, [0, 2, 1])), 1)

                with tf.name_scope("prediction"):
                        self.prediction = tf.nn.softmax(logits, name='prediction')  # 预测值
                        self.MRR_op = mrr(logits, self.Input_label)

                with tf.name_scope("cost"):
                        self.cost = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.Input_label))
                        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
                tf.summary.scalar('loss', self.cost)
                tf.summary.scalar('MRR', self.MRR_op)
                tf.summary.histogram("prediction", self.prediction)
                tf.summary.histogram("Softmax_weight", sampled_W)
                self.merged_summary_op = tf.summary.merge_all()

        def fit(self, iterator, Batch, verbose=100):
                self.sess.run(iterator.initializer)
                summary_writer = tf.summary.FileWriter(self.checkpoint_dir, graph=tf.get_default_graph())
                mean_loss, mean_mrr = np.zeros(verbose), np.zeros(verbose)
                save_step = self.data_size // self.batch_size
                for step in range(20000):
                        try:
                                batch_data = self.sess.run(Batch)
                                feed_dict = {self.Input_Item: batch_data[0],
                                             self.Input_action: batch_data[1],
                                             self.Input_Impression: batch_data[2].reshape(-1, self.impression_length),
                                             self.Input_label: batch_data[3]}
                                idx = step % verbose
                                mean_loss[idx], mean_mrr[idx], summary = self.sess.run(
                                        [self.cost, self.MRR_op, self.merged_summary_op], feed_dict)
                                summary_writer.add_summary(summary, step)

                                if step % verbose == 0:
                                        print(f'Step{step} Loss : {mean_loss.mean()}  MRR:{mean_mrr.mean()}')
                                        mean_loss, mean_mrr = np.zeros(verbose), np.zeros(verbose)
                                if step % save_step == 0:
                                        # saver.save(sess,f'{logs_path}/{MODEL_NAME}')
                                        self.saver.save(self.sess, f'{self.checkpoint_dir}/gru-model', global_step=step,
                                                        write_meta_graph=False)


                        except tf.errors.OutOfRangeError:
                                print("模型训练结束")
                                break


USECOLS = ['session_id', 'action_type', 'reference', 'impressions', 'step']
duplicat_col = ['user_id', u'session_id', u'timestamp', 'reference']


class Doc2VecModel(BaseEstimator, TransformerMixin):
        def __init__(self, vector_size=150, window=10, min_count=1, epochs=None, workers=5):
                self.vector_size = vector_size
                self.window = window
                self.min_count = min_count
                self.epochs = epochs
                self.workers = workers

        def _fit(self, data):
                data = data.drop_duplicates(subset=duplicat_col)
                data['ref_int'] = data.apply(inted_reference, axis=1)
                data = data.query('ref_int==1')[USECOLS]
                self.n_items = data.reference.nunique()
                data_group = data.groupby('session_id')
                self.items_series = []
                self.action_series = []
                for idx, session in tqdm.tqdm(data_group):
                        refs = []
                        acts = []
                        b_step = 0
                        for _, r in session.iterrows():
                                ref = r['reference']
                                step = r['step']
                                act = r['action_type']
                                refs.append(ref)
                                acts.append(act)
                                if step < b_step:
                                        self.items_series.append(refs[:-1])
                                        self.action_series.append(acts[:-1])
                                        refs = copy(refs[-1:])
                                        acts = copy(acts[-1:])

                                        b_step = 0
                                b_step = step
                        self.items_series.append(refs)
                        self.action_series.append(acts)

        def fit_transform(self, X, y=None, **fit_params):
                self._fit(X)
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.items_series)]
                if self.epochs:
                        model = Doc2Vec(documents, vector_size=self.vector_size,
                                        window=self.window,
                                        min_count=self.min_count,
                                        workers=self.workers,
                                        epochs=self.epochs
                                        )

                else:
                        model = Doc2Vec(documents, vector_size=self.vector_size,
                                        window=self.window,
                                        min_count=self.min_count,
                                        workers=self.workers,
                                        )

                self.model = model

        def transform_valid(self, data):
                def check_ref(act, ref):
                        try:
                                int(ref)
                                return 1
                        except:
                                try:
                                        if math.isnan(ref) and (act == 'clickout item'):
                                                return 0
                                        else:
                                                return -1
                                except:
                                        return -1

                data = data.sort_values(by=['session_id', 'timestamp', 'step'])
                data['ref_check'] = data.apply(lambda row: check_ref(row['action_type'], row['reference']), axis=1)
                USECOLS = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions']
                data = data[USECOLS + ['ref_check']]
                data = data.query("ref_check>=0")
                data_group = data.groupby('session_id')
                df = pd.DataFrame()
                for _, session in tqdm.tqdm(data_group):
                        b_step = 0
                        refs = []
                        acts = []
                        for _, r in session.iterrows():
                                ref = r['reference']
                                step = r['step']
                                act = r['action_type']
                                if r['ref_check'] == 0:
                                        r['context_click'] = copy(refs)
                                        r['context_action'] = copy(acts)
                                        df = df.append(r)

                                refs.append(ref)
                                acts.append(act)

                                if step < b_step:
                                        refs = copy(refs[-1:])
                                        acts = copy(acts[-1:])
                                        b_step = 0
                                b_step = step
                cols = ['user_id', 'session_id', 'timestamp', 'step', 'context_click', 'context_action', 'impressions']
                self.valid_data = df[cols]



import matplotlib.pyplot as plt

