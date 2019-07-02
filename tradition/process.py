import sys
sys.path.append('..')
from config import train_path, test_path, item_path
import pandas as pd
from utils import load_pickle, plot_distribution
import tqdm
import numpy as np
import pandas  as pd
from func_utils import stat,group_rank
import  os

class BaseExtactor(object):
        def __init__(self,filename=''):
                self.filename=filename
        def _fit(self,X):
                pass

        def _transform(self,X):
                self._fit(X)

        def fit_transform(self,X):
                self._transform(X).to_pickle(os.path.join(featue_path,self.filename))


class CtrPriceFeature(BaseExtactor):
        '''
        impressions reference prices 特征
        item 的总体统计信息
        1.酒店的price feature for example mean_price std_price
        2.history  exposure info  for example freqs and  rank ([3,2,10] rank=10)
        3.histoy ctr

        '''
        def _fit(self, data):
                click_data = data.query("action_type=='clickout item'")
                click_data = click_data[["impressions", 'prices', 'reference']]
                price_dict = {}
                for i, row in tqdm.tqdm(click_data.iterrows()):
                        click_it = row.reference
                        imp = row.impressions.split('|')
                        _price = list(map(float, row.prices.split('|')))
                        _loc_sess = list(range(len(imp)))
                        for k, p, loc in zip(imp, _price, _loc_sess):
                                if k in price_dict:
                                        price_dict[k]["price"].append(p)
                                        price_dict[k]['location'].append(float(loc))
                                else:
                                        price_dict[k] = {"price": [p], 'location': [float(loc)], 'click_rank': []}

                        if isinstance(click_it, str):
                                if click_it in price_dict:
                                        if click_it in imp:
                                                price_dict[click_it]['click_rank'].append(float(imp.index(click_it)))
                                        else:
                                                price_dict[click_it]['click_rank'].append(-1.)
                                else:
                                        price_dict[click_it] = {"price": [], 'location': [], 'click_rank': [-1.]}

                return price_dict

        def _transform(self, data):
                price_dict = self._fit(data)
                click_data_pl = pd.DataFrame(price_dict).T

                # 传址
                stat(click_data_pl, 'location')
                stat(click_data_pl, 'price')

                click_data_pl['exposure_freq'] = click_data_pl.location.map(len)
                click_data_pl['click_freq_con'] = click_data_pl.click_rank.map(len)
                click_data_pl['click_rank_mean_con'] = click_data_pl.click_rank.map(np.mean).fillna(30.)
                click_data_pl['his_ctr'] = click_data_pl.click_freq_con / click_data_pl.exposure_freq

                return click_data_pl


class LocationFeature(BaseExtactor):
        def _fit(self,loc_data):
                loc_data = loc_data.copy().groupby(['reference', 'city', 'country']).size().reset_index().rename(
                        columns={0: 'freqs'})
                city2id = {}
                for indx, (r, ref) in enumerate(loc_data.groupby('reference')):
                        citys = ref.city.values
                        citys_len = len(citys)
                        if citys_len == 1:
                                city2id.update({citys[0]: indx})
                        else:
                                city2id.update(dict(zip(citys, [indx] * citys_len)))

                country_bag = loc_data.country.unique()
                country2id = dict(zip(country_bag, range(len(country_bag))))
                city2id = pd.Series(city2id).reset_index().rename(columns={'index': 'city', 0: 'city2id'})
                country2id = pd.Series(country2id).reset_index().rename(columns={'index': 'country', 0: 'country2id'})

                return city2id, country2id

        def _transform(self,loc_data):
                city2id, country2id = self._fit(loc_data)
                loc_data = pd.merge(loc_data, city2id, on='city', how='inner')
                loc_data = pd.merge(loc_data, country2id, on='country', how='inner')
                loc_data = loc_data.groupby(['reference', 'city2id', 'country2id']).size().reset_index().rename(
                        columns={0: 'freqs_con'})

                loc_data = loc_data.groupby('city2id').apply(
                        lambda r: group_rank(r, colname='freqs_con', rank_name='city_heat_con'))



                return loc_data




if __name__ == '__main__':
        train_data = pd.read_csv(train_path, nrows=3000)
        test_data = pd.read_csv(test_path, nrows=300)
        data = pd.concat([train_data, test_data], axis=0)
        featue_path='./processed_data'
        # step  CtrPriceFeature
        cpf = CtrPriceFeature(filename='price_rank.pkl')
        click_data_pl = cpf.fit_transform(data)


        # step2 location feature
        digit_cond = data.reference.str.isdigit() == True
        loc_data = data[digit_cond].loc[:, ['reference', 'city']]

        city_conty = loc_data.city.str.split(',')
        loc_data['city'] = city_conty.map(lambda r: r[0])
        loc_data['country'] = city_conty.map(lambda r: r[1])
        lf=LocationFeature(filename='loc_info.pkl')
        loc_data=lf.fit_transform(loc_data)
        # loc_data.to_pickle("./processed_data/price_rank.pkl")
