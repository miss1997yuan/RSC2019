#coding:utf-8
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import Window

GR_COLS = ["user_id", "session_id", "timestamp", "step"]

def transform_trainable(df, test=False):
        '''
        :param df: trivago train set or test set
        :param test: if set is test ,test is True
        :return:
        '''
        inter_list = ['clickout item',
                      'interaction item deals',
                      'interaction item image',
                      'interaction item info',
                      'search for item',
                      'interaction item rating']
        if test:
                inter_list.remove('clickout item')
                df = df.filter(df['action_type'].isin(*inter_list))
        else:
                df = df.filter(df['action_type'].isin(*inter_list))

        df = df.select("session_id", 'reference', ).groupby("session_id").agg(
                F.collect_list('reference').alias('reference_list'))
        return df


@F.udf
def getCosinDis(item, session):
        from scipy.spatial import distance
        if (item == None) | (session == None):
                return 5.
        else:
                return round(float(distance.cosine(item.toArray(), session.toArray())), 5)







class Doc2vec(object):
        def __init__(self,model_path=''):
                self.model_path=model_path

        def fit(self,train_data):
                train_data_seq = transform_trainable(train_data, test=False)

                # step2:Training model and save model
                word2Vec = Word2Vec(vectorSize=100, seed=42, minCount=2, inputCol="reference_list",
                                    outputCol="doc2vec_spark")
                model = word2Vec.fit(train_data_seq)
                model.write().overwrite().save(os.path.join(self.model_path, "item2vec.model"))
                print("The model has been trained")

        def transform(self,train_data,is_test=False):
                model = Word2VecModel.load(self.model_path)
                item2vec = model.getVectors()
                train_data_seq = transform_trainable(train_data, test=is_test)

                train_data_seq = model.transform(train_data_seq)
                # step5:Get the click  sequence
                train_data_click = train_data.filter("action_type='clickout item'").select('user_id',
                                                                                           "session_id",
                                                                                           'timestamp',
                                                                                           'step', 'reference',
                                                                                           'impressions')
                train_data_click = train_data_click.withColumn('impressions',
                                                               F.split(train_data.impressions,
                                                                       '\|')).withColumn("impressions",
                                                                                         F.explode(
                                                                                                 "impressions"))

                cond = train_data_click.impressions == item2vec.word
                df_out = train_data_click.join(item2vec, cond, how='left').select('user_id', "session_id",
                                                                                  'timestamp', 'step',
                                                                                  'reference', 'impressions',
                                                                                  'vector')
                df_out = df_out.join(train_data_seq, df_out.session_id == train_data_seq.session_id,
                                     how='left').drop(
                        train_data_seq.session_id)

                # step6: Find and sort the similarity between the session vector and the exposure vector
                df_out = df_out.withColumn('sim', getCosinDis('vector', 'item2vec'))
                df_out = df_out.withColumn("sim", df_out.sim.cast('float')).withColumn(
                        "rank",
                        F.rank().over(Window.partitionBy("session_id", 'timestamp', "step").orderBy("sim")))


                return df_out
