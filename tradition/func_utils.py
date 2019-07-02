import numpy as np


def stat(df, colname=''):
        df[colname + '_mean_con'] = df[colname].map(lambda r: None if len(r) == 0 else np.mean(r))
        df[colname + '_min_con'] = df[colname].map(lambda r: None if len(r) == 0 else np.min(r))
        df[colname + '_max_con'] = df[colname].map(lambda r: None if len(r) == 0 else np.max(r))
        df[colname + '_std_con'] = df[colname].map(lambda r: None if len(r) == 0 else np.std(r))


def group_rank(group, colname='', rank_name=''):
        max_rank = group[colname].nunique()
        group[rank_name] = group[colname].rank(method='dense', ascending=False) / float(max_rank)
        return group
