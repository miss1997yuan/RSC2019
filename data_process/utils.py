import pandas as pd

# Test
test_headder='user_id,session_id,timestamp,step,action_type,reference,platform,city,device,current_filters,impressions,prices,check_ref,day,day_month,timestamp_context,step_context,action_type_context,reference_context,platform_context,city_context,device_context,current_filters_context'

def get_sess_context(sess):
        if sess.shape[0] == 1:
                for c in to_listcol:
                        sess[c + '_context'] = ''
                return sess
        n_day = sess.day.nunique()
        #  一个sess 中有多少个预测对象以及相应day
        checkNaN = sess.query("check_ref==1 and action_type=='clickout item'").day.values
        n_NaN = checkNaN.shape[0]

        if n_NaN == 1:
                day_sess = sess.query("day==%d" % checkNaN[0])
                con_feat = context_feature(day_sess)
        else:
                con_feat = pd.DataFrame()
                for d in checkNaN:
                        day_sess = sess.query("day==%d" % d)
                        con1 = context_feature(day_sess)
                        con_feat = con_feat.append(con1)
        return con_feat


def context_feature(sess):
        check0 = sess.query('check_ref==0')
        check1 = sess.query("check_ref==1 and action_type=='clickout item'")
        check0=check0.fillna(' ')
        for c in to_listcol:

              check1[c + '_context'] = '|'.join(check0[c].astype(str).values)
        return check1


def addflag(df):
        df['check_ref'] = df.reference.isnull().astype(int)
        # 一个session 中重复的step
        df['day'] = df.timestamp.map(lambda line: pd.Timestamp(line, unit='s').dayofweek)
        df['day_month'] = df.timestamp.map(lambda line: pd.Timestamp(line, unit='s').day)
        return df


to_listcol = ['timestamp', 'step', 'action_type', 'reference', 'platform', 'city', 'device', 'current_filters']
USECOLS = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions', ]
duplicat_col = ['user_id', u'session_id', u'timestamp', 'reference']
