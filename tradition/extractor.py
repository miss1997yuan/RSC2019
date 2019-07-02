import sys
sys.path.append('..')
from config  import train_path,test_path,item_path
import pandas as pd
from utils  import inted_reference,to_pickle
from collections  import Counter
import tqdm
import argparse
from  copy import copy
USECOLS = ['user_id','session_id', 'timestamp', 'step','action_type', 'reference', 'impressions', ]
duplicat_col = ['user_id', u'session_id', u'timestamp', 'reference']
counter=lambda line:dict(Counter(line))

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


def ExtractorTriain(data):
        data['ref_int'] = data.apply(inted_reference, axis=1)
        data = data[USECOLS]
        data_group = data.groupby('session_id')
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
                        if act == 'clickout item':
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
        df = data.query("action_type=='clickout item'")[
                ['user_id', 'session_id', 'timestamp', 'step', 'reference', 'impressions']]
        df['item_series'] = items_series
        df['action_series'] = action_series
        df['action_series_map'] = action_series_map
        df['it_map'] = [counter(zip(*line)) for line in zip(items_series, action_series)]

        df = explode_colum(df)
        df['impress_action'] = df.apply(lambda row: getImpMap(row['it_map'], row['impression']), axis=1)
        df.to_csv('./dealed_data/sess_data/sess_context.csv', mode='a+', header=False)

        return df



if  __name__=='__main__':
        args=argparse.ArgumentParser("extractor session context feature")
        args.add_argument('--train',type=int,default=1,help='train==1 else 0')
        args.add_argument('--name',type=str,default='session_context',help='file name pkl ')
        params=args.parse_args()
        if params.train!=1:
                train_data = pd.read_csv(train_path, nrows=3000)
        else:
                train_data = pd.read_csv(train_path, nrows=3000)

        deal_train = ExtractorTriain(train_data)

        col = ['reference', 'action_series_map', 'impress_action', 'impression', 'impress_rank_cat']
        deal_train = deal_train[col]

        action_series_map = pd.DataFrame.from_records(deal_train.action_series_map).fillna(0.0)
        impress_action = pd.DataFrame.from_records(deal_train.impress_action).fillna(0.0)
        train_data = pd.concat([deal_train[['reference', 'impression', 'impress_rank_cat']],
                                action_series_map, impress_action], axis=1)

        # to_pickle(train_data,)



