import pandas as pd
import copy
import tqdm
from utils import  inted_reference


USECOLS=['session_id','action_type','reference','impressions','step']

duplicat_col=['user_id', u'session_id', u'timestamp','reference']


class Doc2VecExtractor(object):
        def fit(self,data):
                data = data.drop_duplicates(subset=duplicat_col)
                data['ref_int'] = data.apply(inted_reference, axis=1)
                data = data.query('ref_int==1')[USECOLS]
                self.n_items=data.reference.nunique()
                data_group = data.groupby('session_id')
                items_series = []
                for idx, session in tqdm.tqdm(data_group):
                        refs = []
                        b_step = 0
                        for _, r in session.iterrows():
                                ref = r['reference']
                                step = r['step']
                                refs.append(ref)
                                if step < b_step:
                                        items_series.append(refs[:-1])
                                        refs = copy.copy(refs[-1:])
                                        b_step = 0
                                b_step = step
                        items_series.append(refs)

from sklearn.base import  BaseEstimator,TransformerMixin

class Doc2VecModel(BaseEstimator,TransformerMixin):
        def fit_transform(self, X, y=None, **fit_params):
                pass












