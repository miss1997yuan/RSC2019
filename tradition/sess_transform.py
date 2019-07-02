import sys
sys.path.append('..')
from config import train_path, test_path, item_path
import pandas as pd
from config import actions,interaction_item
import  tqdm
actions_map=dict(zip(actions,[0.]*10))
interaction_map=dict(zip(interaction_item,[0.0]*6))

sess_feature_name = ['change of sort order', 'clickout item', 'filter selection',
                             'interaction item deals', 'interaction item image',
                             'interaction item info', 'interaction item rating',
                             'search for destination', 'search for item', 'search for poi',
                             'clickout item_it', 'interaction item deals_it',
                             'interaction item image_it', 'interaction item info_it',
                             'interaction item rating_it', 'search for item_it']
def item_action(group):
        item_act_map = interaction_map.copy()
        item_act_map.update(group['action_type'].value_counts().to_dict())
        return item_act_map
#  explode
def sess_explode(sess):
        def action_ext(one_click):

                acts_map = actions_map.copy()
                acts = one_click.action_type.value_counts().to_dict()
                acts_map.update(acts)

                cond = one_click.reference.str.isdigit()
                _it_act = one_click[cond].dropna(subset=['reference']).loc[:, ['action_type', 'reference']].groupby('reference').apply(item_action)
                _it_act = pd.DataFrame(_it_act.to_dict()).T.add_suffix('_it')
                return acts_map, _it_act

        click_data = sess.query("action_type=='clickout item'")
        uniqe_imp = click_data.loc[:, ['timestamp', 'impressions']]
        uniqe_imp = pd.DataFrame(uniqe_imp.impressions.str.split('|').tolist(), index=uniqe_imp.timestamp).stack()
        uniqe_imp.columns = ['timestamp', 'impressions']
        uniqe_imp = uniqe_imp.reset_index().rename(columns={0: 'impression', 'level_1': 'impress_rank_cat'})
        #         uniqe_imp = uniqe_imp.reset_index([0, 'timestamp']).rename(columns={0:'impression'})
        sess_exp = pd.merge(click_data.iloc[:, :9], uniqe_imp, on='timestamp')
        times = click_data.timestamp.values
        steps = click_data.step.values
        #  action feature

        acts_f = {}
        it_act = pd.DataFrame()
        if len(times) == 1:
                sess = sess.query("action_type!='clickout item'")
                acts_map, _it_act = action_ext(sess)
                acts_f[times[0]] = acts_map
                it_act = _it_act
        else:
                for t, st in zip(times, steps):
                        if st > 1:
                                one_click = sess.query("timestamp<%d" % t)
                                if one_click.shape[0] > 0: # click time equal
                                        acts_map, _it_act = action_ext(one_click)
                                        acts_f[t] = acts_map
                                        # item->action feature
                                        it_act = it_act.append(_it_act)
        sess_act = pd.DataFrame(acts_f).T
        sess_exp = pd.merge(sess_exp, sess_act, left_on='timestamp', right_index=True)
        sess_exp = pd.merge(sess_exp, it_act, left_on='impression', right_index=True, how='left')
        sess_exp = sess_exp.fillna(0.0)
        sess_exp['hour'] = sess_exp.timestamp.map(lambda ts: pd.Timestamp(ts, unit='s').hour)
        sess_exp['day_of_week'] = sess_exp.timestamp.map(lambda ts: pd.Timestamp(ts, unit='s').dayofweek)

        return sess_exp

def transform_less1(data_less1):
        #     sess=pd.DataFrame(data_less1.iloc[0,:]).T
        sess = data_less1
        uniqe_imp = sess.loc[:, ['timestamp', 'impressions']]

        uniqe_imp = pd.DataFrame(uniqe_imp.impressions.str.split('|').tolist(), index=uniqe_imp.timestamp).stack()

        uniqe_imp.columns = ['timestamp', 'impressions']
        uniqe_imp = uniqe_imp.reset_index().rename(columns={0: 'impression', 'level_1': 'impress_rank_cat'})

        # uniqe_imp = uniqe_imp.reset_index([0, 'timestamp']).rename(columns={0:'impression'})

        sess_exp = pd.merge(sess.iloc[:, :9], uniqe_imp, on='timestamp')
        nums = sess_exp.shape[0]
        app = pd.DataFrame(dict(zip(sess_feature_name, [0.] * len(sess_feature_name))), index=range(nums))

        sess_exp = pd.concat([sess_exp, app], axis=1)
        sess_exp['hour'] = sess_exp.timestamp.map(lambda ts: pd.Timestamp(ts, unit='s').hour)
        sess_exp['day_of_week'] = sess_exp.timestamp.map(lambda ts: pd.Timestamp(ts, unit='s').dayofweek)

        return sess_exp


def main(Train='train'):
        if Train=='train':
                data = pd.read_csv(train_path, nrows=1000)
        else:
                data = pd.read_csv(test_path, nrows=3000)
        data.sort_values(by=['session_id', 'timestamp', 'step'], inplace=True)
        click_in_sess = data.query("action_type=='clickout item'").session_id.drop_duplicates()
        data = pd.merge(data, click_in_sess.reset_index(), on='session_id')
        sess_size = data.groupby('session_id').size()
        greater1 = sess_size[sess_size > 2].reset_index()
        less1 = sess_size[sess_size == 1].reset_index()
        datag1 = pd.merge(data, greater1, on='session_id').iloc[:, :-3]
        dataL1 = pd.merge(data, less1, on='session_id').iloc[:, :-3]

        step_click1 = data.query("action_type=='clickout item' and step==1")
        cols = dataL1.columns
        dataL1 = pd.concat([step_click1[cols], dataL1])
        if dataL1.shape[0] > 1:
                dataL1 = transform_less1(dataL1)

        dfG1 = pd.DataFrame()
        gg = datag1.groupby('session_id')
        for _, sess in tqdm.tqdm(gg):
                dfG1 = dfG1.append(sess_explode(sess))

        #         df = pd.concat([dfG1, dataL1])

        #         df = pd.concat([dfG1, dataL1])
        dataL1.to_pickle("./processed_data/sess_sizeLess1_%s.pkl"%Train)
        dfG1.to_pickle("./processed_data/sess_size_Greater1_%s.pkl"%Train)


if __name__=='__main__':
        main('test')