import pandas as pd
from utils import inted_reference, to_pickle
from copy import copy
import time
import argparse
import keras.preprocessing.sequence  as kps
import tensorflow as tf
import numpy as np

USECOLS = ['session_id', 'action_type', 'reference', 'impressions', 'step']
duplicat_col = ['user_id', u'session_id', u'timestamp', 'reference']

from sklearn.base import TransformerMixin


class GRUExtractor(TransformerMixin):
        def __init__(self, min_sess=3, max_item_seq=30):
                self.min_sess = min_sess
                self.max_item_seq = max_item_seq

        def _fit(self, data):
                data = data.drop_duplicates(subset=duplicat_col)
                data.sort_values(by=['session_id', 'timestamp', 'step'], inplace=True)
                data['ref_int'] = data.apply(inted_reference, axis=1)
                data = data.query('ref_int==1')[USECOLS]
                action_unqiue = data.action_type.unique().tolist()
                self.action_map = dict(zip(action_unqiue, range(len(action_unqiue))))
                data['action_type'] = data.action_type.apply(lambda act: self.action_map[act])
                impression_list = data.impressions.dropna().str.split('|').values.tolist()
                impression_set = set([])
                for m in impression_list:
                        impression_set.update(m)
                reference_list = data.reference.values.tolist()
                impression_set.update(reference_list)

                self.ref2id = dict(zip(impression_set, range(len(impression_set))))

                data['reference'] = data.reference.apply(lambda ref: self.ref2id[ref])
                self.id2ref = dict(zip(self.ref2id.values(), self.ref2id.keys()))
                self.click_id = self.action_map['clickout item']
                self.n_items = len(self.ref2id)
                session_size = data.groupby('session_id').size()

                session_size = session_size[session_size >= self.min_sess].reset_index()
                data = pd.merge(data, session_size, on='session_id', how='inner').iloc[:, :-1]

                return data

        def _transform(self, data):
                encode_impressions = lambda imp: map(lambda r: self.ref2id[r], imp)
                data_group = data.groupby('session_id')
                session_id = []
                self.item_series = []
                self.action_series = []
                self.impression_series = []
                self.labels = []
                self.item_series_length = 0
                for idx, session in data_group:
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
                                refs.append(ref)
                                acts.append(act)
                                if act == self.click_id:

                                        if len(refs) > 1:
                                                session_id.append(r['session_id'])
                                                self.labels.append([ref])
                                                self.impression_series.append(
                                                        list(encode_impressions(imp.split("|"))))
                                                if len(refs) > self.max_item_seq:

                                                        self.item_series.append(copy(refs[-self.max_item_seq - 1:-1]))
                                                        self.action_series.append(copy(acts[-self.max_item_seq - 1:-1]))
                                                else:
                                                        self.item_series.append(copy(refs[:-1]))
                                                        self.action_series.append(copy(acts[:-1]))

                self.data_size = len(self.labels)

                # self.process_data={'item_series': item_series, 'action_series': action_series,
                #            'impression_series': impression_series,
                #            'labels': labels}

        def fit_transform(self, X, y=None, **fit_params):
                data = self._fit(X)
                self._transform(data)
                print('Congratulations on your data processing success')
                self.info = f'''action map:{self.action_map} \n  n_items:{self.n_items}\n data size is {self.data_size}
                max item length {self.max_item_seq}'''
                print(self.info)


class Paded(object):
        def __init__(self, impression_length=25):
                self.impression_length = impression_length

        def _fit(self, extractor):
                self.n_items = extractor.n_items
                self.item_series_length = extractor.max_item_seq
                self.n_action = len(extractor.action_map)
                self.click_item = extractor.labels
                self.impression_series = extractor.impression_series
                self.action_series = extractor.action_series
                self.item_series = extractor.item_series
                self.data_size = extractor.data_size

        def _transform(self):
                labels = []
                for cl, im in zip(self.click_item, self.impression_series):
                        iml = len(im)
                        diffs = self.impression_length - iml
                        pad = [im[np.random.randint(0, iml)] for _ in range(diffs)]
                        im += pad
                        labels.append((np.array(im) == cl[0]).astype(int))
                self.labels = np.array(labels)
                self.action_series = kps.pad_sequences(self.action_series, self.item_series_length, padding='post')
                self.item_series = kps.pad_sequences(self.item_series, self.item_series_length, padding='post')
                self.impression_series = np.array(self.impression_series)

        def _Iterator(self):
                data = (self.item_series, self.action_series, self.impression_series, self.labels)
                dataset = tf.data.Dataset.from_tensor_slices(data)
                dataset = dataset.repeat(self.epoches).batch(self.batch_size).shuffle(buffer_size=2000)
                iterator = dataset.make_initializable_iterator()
                return iterator, iterator.get_next()

        def fit_transform(self, extractor, epoches=30, batch_size=100):
                self.batch_size = batch_size
                self.epoches = epoches
                self._fit(extractor)
                self._transform()
                return self._Iterator()


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_item_seq', type=int, default=30, help='max_item_seq')
        args = parser.parse_args()
        max_item_seq = args.max_item_seq
        print('--*--' * 10)
        print(f'Max Item Sequnce :{max_item_seq}')
        start = time.time()
        train_path = '../datasets/train.csv'
        test_path = '../datasets/test.csv'
        train_data = pd.read_csv(train_path, nrows=3000)
        test_data = pd.read_csv(test_path, nrows=3000)
        data = pd.concat([train_data, test_data], axis=0)
        grudata = GRUExtractor(min_sess=3)
        grudata.fit_transform(X=data)
        data_name = f'./processed_data/gru_data_shift_{max_item_seq}.pkl'
        to_pickle(grudata, data_name)

        pp = Paded()
        pp.fit_transform(grudata)
        to_pickle(pp, data_name + "t")
        end = time.time()
        print(f"Data processing time :{end-start}")
