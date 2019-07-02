import tensorflow as tf

import keras
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import Window

from pyspark.sql import SparkSession
from pyspark import SparkConf
import os

from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.sql.types import ArrayType, StringType, IntegerType
import pyspark.sql.functions  as F
import numpy as np

model_pth = '/team/cmp/hive_db/cmp_tmp/dl_model_template/Recsys/wyd/models'
model_name = 'weight_dssm_cvmodel.model'


@F.udf(returnType=ArrayType(StringType()))
def item_seq(item_list, impression):
        return item_list + [impression]


@F.udf(returnType=ArrayType(IntegerType()))
def item_list2id(item_list):
        if len(item_list) > 0:
                map_ = bitem2id.value
                items = []
                for line in item_list:
                        if line in map_:
                                items.append(map_[line])
                return items

        else:
                return []


@F.udf(returnType=ArrayType(IntegerType()))
def action_list2id(action_list):
        if len(action_list) > 0:
                map_ = baction2id.value
                acts = []
                for line in action_list:
                        if line in map_:
                                acts.append(map_[line])
                return acts

        else:
                return []


@F.udf(returnType=IntegerType())
def impression2id(r):
        map_ = bitem2id.value
        if r in map_:
                return map_[r]
        else:
                return -1


class Data_Generation(object):

        def __init__(self, sqlContext, test=True, has_cv=True, minImpressionFreqs=5):
                self.sqlContext = sqlContext
                self.test = test
                self.minImp = minImpressionFreqs
                self.has_cv = has_cv

        def _fit(self):
                sqlContext.sql("use cmp_tmp")

                if self.test:
                        data = sqlContext.sql(
                                "select * from cmp_tmp_rec_test_agg where action_type=='clickout item'  and  reference_list is not  NULL")
                else:
                        data = sqlContext.sql(
                                "select * from cmp_tmp_rec_train_agg where reference_list is not NULL ")

                item_info = sqlContext.sql("select * from cmp_tmp_rec_item_info_feature ")
                item_info = item_info.filter("impression_freqs>%d" % self.minImp)

                data = data.join(item_info, data.impression == item_info.item_id, how='inner')
                data = data.withColumn("time_diff_list", F.split("time_diff_list", "\|"))
                digit_ref = F.udf(lambda arr: [line for line in arr if line.isdigit()],
                                  returnType=ArrayType(StringType()))
                action_ref = F.udf(lambda arr, acts: [act_ for line, act_ in zip(arr, acts) if line.isdigit()],
                                   returnType=ArrayType(StringType()))

                data = data.withColumn('item_list',
                                       digit_ref('reference_list'))

                data = data.withColumn('action_list',
                                       action_ref('reference_list', 'action_type_list')).withColumn(
                        'time_diff_list',
                        action_ref('reference_list', 'time_diff_list'))

                data = data.withColumn("label", (data.reference == data.impression).cast('int')).withColumn(
                        "item_seq", item_seq("item_list", 'impression'))

                return data

        def get_cv_model(self):
                if self.has_cv:
                        from pyspark.ml.feature import CountVectorizerModel
                        cv_model = CountVectorizerModel.load(os.path.join(model_pth, model_name))
                else:
                        from pyspark.ml.feature import CountVectorizer
                        data = self._fit()
                        cv = CountVectorizer(inputCol='item_seq', outputCol='item_seq_enc', vocabSize=1 << 20,
                                             minTF=0, minDF=0)
                        cv_model = cv.fit(data)
                        cv_model.write().overwrite().save(os.path.join(model_pth, model_name))

                copora = cv_model.vocabulary  # 579012
                action_copora = ['clickout item',
                                 'interaction item deals',
                                 'interaction item image',
                                 'interaction item info',
                                 'search for item',
                                 'interaction item rating']
                item2id = dict(zip(copora, range(1, len(copora) + 1)))
                action2id = dict(zip(action_copora, range(1, len(action_copora) + 1)))
                sc = self.sqlContext.sparkContext
                bitem2id = sc.broadcast(item2id)
                baction2id = sc.broadcast(action2id)
                print("Item size:", len(item2id))
                return bitem2id, baction2id

        def transform(self):
                data = self._fit()
                data = data.withColumn("item_list", item_list2id('item_list'))

                data = data.withColumn("impression2id",
                                       impression2id('impression')).withColumn(
                        "action_list", action_list2id('action_list')).withColumn('impressions',
                                                                                 item_list2id('impressions'))

                GR_COLS = ['user_id', 'session_id', 'timestamp', 'step', 'item_list', 'action_list',
                           'time_diff_list',
                           'impression', 'impressions', 'impression2id', 'label', 'flag', 'impress_rank',
                           'price_rank']
                data = data.select(*GR_COLS)

                return data


if __name__ == '__main__':
        # pth = " --master yarn    pyspark-shell"
        #
        # os.environ['PYSPARK_SUBMIT_ARGS'] = pth

        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.executor.memory', '4g'),
                                   ('spark.executor.cores', '4'), ('spark.driver.memory', '10G'),
                                   ('spark.sql.shuffle.partitions', '100'),
                                   ('spark.cores.max', '4'), ('spark.executor.instances', '100'),
                                   ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()

        #  train set  vocabul:475285
        data_func = Data_Generation(sqlContext=sqlContext, test=False, has_cv=True, minImpressionFreqs=10)
        bitem2id, baction2id = data_func.get_cv_model()
        train_data = data_func.transform()
        # test set
        data_func = Data_Generation(sqlContext=sqlContext, test=True, has_cv=True, minImpressionFreqs=10)
        test_data = data_func.transform()

        #  Data output
        train_data.drop('impressions').filter('size(item_list)>0'
                                              ).limit(100000).toPandas().to_csv('weight_dssm/dssm_train-demo.csv',
                                                                                index=False)

        train_data.drop('impressions').filter('size(item_list)>0'
                                              ).toPandas().to_csv('weight_dssm/dssm_train.csv', index=False)
        test_data.drop('impressions').filter('size(item_list)>0'
                                             ).toPandas().to_csv('weight_dssm/dssm_test.csv', index=False)
        # step 3: data output
        uniq_keys = ['user_id', 'session_id', 'timestamp', 'step']
        train_cols = ['item_list', 'action_list', 'impression2id', 'label']
        test_cols = ['item_list', 'action_list', 'impression2id', 'impression', 'label']

        data_path = 'weight_dssm'
        join_path = lambda pth: os.path.join(data_path, pth)

        # step 4:dssm
        # train data
        train_data.selectExpr(*train_cols).filter('size(item_list)>0').filter("flag=='train'").toPandas().to_csv(
                join_path('dssm_train.csv'), index=False)
        # test data
        train_data.selectExpr(*train_cols).filter('size(item_list)>0').filter("flag=='test'").toPandas().to_csv(
                join_path('dssm_test.csv'), index=False)

        # valid data
        train_data.selectExpr(*(uniq_keys + train_cols)).filter('size(item_list)>0').filter(
                "flag=='test'").toPandas().to_pickle(
                join_path('dssm_valid.pkl'))
        # submission data
        sub_mission = test_data.selectExpr(*(uniq_keys + test_cols)).filter('size(item_list)>0').toPandas().to_pickle(
                join_path('dssm_submission.pkl'))  # 160651
        # step 5 nocontext

        # -------------------impress locmodel --------*----------------------------

        data_path = '/team/cmp/hive_db/cmp_tmp/dl_model_template/Recsys/wyd/loc_impress/data'

        join_path = lambda pth: os.path.join(data_path, pth)

        impress_train_cols = ['impressions', 'impression2id', 'impress_rank', 'label']

        uniq_keys = ["user_id", "session_id", "timestamp", "step"]
        train_data.filter("flag=='train'"
                          ).selectExpr(impress_train_cols
                                       ).write.parquet(join_path('impress_train'), mode='overwrite')

        train_data.filter("flag=='test'"
                          ).filter('label==1').selectExpr(impress_train_cols
                                                          ).write.parquet(join_path('impress_test'), mode='overwrite')

        train_data.filter("flag=='test'"
                          ).selectExpr(*(uniq_keys + impress_train_cols)
                                       ).write.parquet(join_path('impress_valid'), mode='overwrite')
