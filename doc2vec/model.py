import sys
sys.path.append('..')
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from copy import copy
from utils import *
import pandas as pd
import argparse
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
                index2word = model.wv.index2word
                n_items = len(index2word)
                index = np.arange(1, n_items + 1)
                self.item2id = dict(zip(index2word, index))

        def transform_valid(self, data):
                def check_ref(act, ref):
                        if isinstance(ref, float):
                                if (act == 'clickout item'):
                                        return 0
                                else:
                                        return -1
                        else:
                                try:
                                        int(ref)
                                        return 1
                                except:
                                        return -1

                data = data.drop_duplicates(subset=duplicat_col)
                data = data.sort_values(by=['session_id', 'timestamp', 'step'])
                data['ref_check'] = data.apply(lambda row: check_ref(row['action_type'], row['reference']), axis=1)
                USECOLS = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions']
                data = data[USECOLS + ['ref_check']]
                data = data.query("ref_check>=0")
                data_group = data.groupby('session_id')
                item_series = []
                action_series = []

                for _, session in tqdm.tqdm(data_group):
                        b_step = 0
                        refs = []
                        acts = []
                        for _, r in session.iterrows():
                                ref = r['reference']
                                step = r['step']
                                act = r['action_type']
                                if r['ref_check'] == 0:
                                        item_series.append(copy(refs))
                                        action_series.append(copy(acts))

                                refs.append(ref)
                                acts.append(act)

                                if step < b_step:
                                        refs = copy(refs[-1:])
                                        acts = copy(acts[-1:])
                                        b_step = 0
                                b_step = step

                df = data.query('ref_check==0')[['user_id', 'session_id', 'timestamp', 'step', 'impressions']]
                df['context_items'] = item_series
                df['context_actions'] = action_series
                df['context_items_idx'] = [list(map(lambda r: self.item2id.get(r), line)) for line in item_series]
                df['context_actions_idx'] = [list(map(lambda r: self.item2id.get(r), line)) for line in action_series]
                self.valid_data=df


import numpy as np


class SeqTrain(object):
        def __init__(self, doc2vecmodel=None, item2id=None):
                self.doc2vecmodel = doc2vecmodel
                self.item2id = item2id

        def _fit(self):
                if self.item2id:
                        self.id2item = dict(zip(self.item2id.values(), self.item2id.keys()))
                        self.n_items = len(self.item2id)
                else:
                        index2word = self.doc2vecmodel.wv.index2word
                        self.n_items = len(index2word)
                        index = np.arange(1, self.n_items + 1)
                        self.item2id = dict(zip(index2word, index))
                        self.id2item = dict(zip(index, index2word))

        def transform(self, X):
                def inted_reference(series):
                        try:
                                int(series)
                                return 1
                        except:
                                return 0

                self._fit()
                data = X.drop_duplicates(subset=duplicat_col)
                data = data.sort_values(by=['session_id', 'timestamp', 'step'])
                data['ref_check'] = data.reference.map(inted_reference)
                data = data.query('ref_check==1')
                actions = data.action_type.unique()
                self.action2id = dict(zip(actions, np.arange(len(actions))))
                self.id2action = dict(zip(np.arange(len(actions)), actions))
                click_id = self.action2id['clickout item']
                data.action_type = data.action_type.map(lambda r: self.action2id[r])
                USECOLS = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions']
                data = data[USECOLS + ['ref_check']]
                data.reference = data.reference.map(lambda r: self.item2id.get(r, 0))
                data_group = data.groupby('session_id')

                self.item_series = []
                self.action_series = []
                self.impression_series = []
                self.labels = []

                for idx, session in tqdm.tqdm(data_group):
                        refs = []
                        acts = []
                        b_step = 0
                        for _, r in session.iterrows():
                                act = r['action_type']
                                ref = r['reference']
                                imp = r['impressions']
                                step = r['step']

                                if step < b_step:
                                        refs = []
                                        acts = []
                                        b_step = 0
                                refs.append(ref)
                                acts.append(act)
                                b_step = step
                                if act == click_id:
                                        if len(refs) > 1:
                                                self.labels.append([ref])
                                                imp = list(map(lambda r: self.item2id.get(r, 0), imp.split('|')))
                                                self.impression_series.append(imp)
                                                self.item_series.append(copy(refs[:-1]))
                                                self.action_series.append(copy(acts[:-1]))


if __name__ == '__main__':

        train_path = '../../datasets/train.csv'
        test_path = '../../datasets/test.csv'
        # train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path, nrows=3000)
        # data = pd.concat([train_data, test_data], axis=0)
        doc2vec_model = Doc2VecModel(vector_size=150, window=5, min_count=1, epochs=None, workers=5)
        doc2vec_model.fit_transform(test_data)
        doc2vec_model.transform_valid(test_data)

        docmodel = doc2vec_model.model
        action_series = doc2vec_model.action_series
        items_series = doc2vec_model.items_series

        train = SeqTrain(item2id=doc2vec_model.item2id)
        train.transform(test_data)
        # print(train.item_series)
