import tqdm
import sys
sys.path.append('..')
from config import train_path, test_path, item_path
import pandas as pd
from collections import Counter

from copy import copy

USECOLS = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions', ]
duplicat_col = ['user_id', u'session_id', u'timestamp', 'reference']


def explode_colum(click_data):
        uniqe_imp = click_data.loc[:, ['timestamp', 'impressions']]
        uniqe_imp = pd.DataFrame(uniqe_imp.impressions.str.split('|').tolist(), index=uniqe_imp.timestamp).stack()
        uniqe_imp.columns = ['timestamp', 'impressions']
        uniqe_imp = uniqe_imp.reset_index().rename(columns={0: 'impression', 'level_1': 'impress_rank_cat'})
        #         uniqe_imp = uniqe_imp.reset_index([0, 'timestamp']).rename(columns={0:'impression'})
        sess_exp = pd.merge(click_data, uniqe_imp, on='timestamp')
        return sess_exp


def getImpMap(it_map, imp):
        sess_map = {}
        for (ref, act), v in it_map.items():
                if ref == imp:
                        sess_map.update({act + '_it': v})
        return sess_map


def ExtractorTest(data):
        def check_ref(act, ref):
                if isinstance(ref, float):
                        if act == 'clickout item':
                                return 0
                        else:
                                return -1
                else:
                        try:
                                int(ref)
                                return 1
                        except:
                                return -1

        #     data['ref_int'] = data.apply(inted_reference, axis=1)
        data['ref_check'] = data.apply(lambda row: check_ref(row['action_type'], row['reference']), axis=1)
        # data = data.query('ref_int==1')[USECOLS]
        data_group = data.groupby('session_id')
        counter = lambda line: dict(Counter(line))
        items_series = []
        action_series = []
        action_series_map = []
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
                        if r['ref_check'] == 0:
                                items_series.append(copy(refs[:-1]))
                                action_series.append(copy(acts[:-1]))
                                action_series_map.append(counter(copy(acts[:-1])))
                        if step < b_step:
                                items_series.append(refs[:-1])
                                action_series.append(acts[:-1])
                                action_series_map.append(counter(copy(acts[:-1])))
                                refs = copy(refs[-1:])
                                acts = copy(acts[-1:])

                                b_step = 0
                        b_step = step

        df = data.query("ref_check==0")  # [['user_id', 'session_id', 'timestamp', 'step','reference', 'impressions']]
        df['item_series'] = items_series
        df['action_series'] = action_series
        df['action_series_map'] = action_series_map
        df['it_map'] = [counter(zip(*line)) for line in zip(items_series, action_series)]

        df = explode_colum(df)
        df['impress_action'] = df.apply(lambda row: getImpMap(row['it_map'], row['impression']), axis=1)

        df = transform_test(df)
        return df


def transform_test(deal_train):
        deal_train['hour'] = deal_train.timestamp.map(lambda r: pd.Timestamp(r, unit='s').hour)

        action_series_map = pd.DataFrame.from_records(deal_train.action_series_map).fillna(0.0)
        impress_action = pd.DataFrame.from_records(deal_train.impress_action).fillna(0.0)
        train_df = pd.concat([deal_train, action_series_map, impress_action], axis=1)

        df = train_df.drop(['reference', 'impressions', 'item_series',
                            'action_series', 'action_series_map',
                            'it_map', 'impression', 'impress_action', 'action_type', 'platform',
                            'city', 'device', 'current_filters', 'prices', 'ref_check'], axis=1)
        return df

if __name__=='__main__':
        test_data=pd.read_csv(test_path)
        deal_train=ExtractorTest(test_data)
        deal_train.to_csv(".test_train.csv")
