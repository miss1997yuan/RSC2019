# coding:utf-8
from functions import *
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import Window

import tensorflow as tf


if __name__ == '__main__':
        # pth = " --master yarn    pyspark-shell"

        # os.environ['PYSPARK_SUBMIT_ARGS'] = pth

        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.app.name', 'rec'), ('spark.executor.memory', '3g'),
                                   ('spark.executor.cores', '4'), ('spark.sql.shuffle.partitions', '300'),
                                   ('spark.cores.max', '6'), ('spark.executor.instances', '100'),
                                   ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()

        sqlContext.sql('use cmp_tmp')
        train_data = sqlContext.sql("select * from cmp_tmp_rec_train_data where flag='train'")

        # step1:convert  item_id to  sequence  in session
        train_data_seq = transform_trainable(train_data, test=False)

        # step2:Training model and save model
        word2Vec = Word2Vec(vectorSize=100, seed=42, minCount=2, inputCol="reference_list", outputCol="doc2vec_spark")
        model = word2Vec.fit(train_data_seq)

        model_path = '/team/cmp/hive_db/cmp_tmp/dl_model_template/recdata/rec_models'
        model.write().overwrite().save(os.path.join(model_path, "item2vec.model"))

        # step3:load model
        model = Word2VecModel.load(os.path.join(model_path, "item2vec.model"))
        # step4:Get the vector for each item and session

        item2vec = model.getVectors()
        train_data_seq = model.transform(train_data_seq)

        # step5:Get the click  sequence
        train_data_click = train_data.filter("action_type='clickout item'").select('user_id', "session_id", 'timestamp',
                                                                                   'step', 'reference', 'impressions')
        train_data_click = train_data_click.withColumn('impressions',
                                                       F.split(train_data.impressions, '\|')).withColumn("impressions",
                                                                                                         F.explode(
                                                                                                                 "impressions"))

        cond = train_data_click.impressions == item2vec.word
        df_out = train_data_click.join(item2vec, cond, how='left').select('user_id', "session_id", 'timestamp', 'step',
                                                                          'reference', 'impressions', 'vector')
        df_out = df_out.join(train_data_seq, df_out.session_id == train_data_seq.session_id, how='left').drop(
                train_data_seq.session_id)

        # step6: Find and sort the similarity between the session vector and the exposure vector
        df_out = df_out.withColumn('sim', getCosinDis('vector', 'item2vec'))
        df_out = df_out.withColumn("sim", df_out.sim.cast('float')).withColumn(
                "rank", F.rank().over(Window.partitionBy("session_id", 'timestamp', "step").orderBy("sim")))

        # step7 loss mrr
        recommand_cord = df_out.filter(df_out.session_id.isNotNull()).select(
                "user_id", 'session_id', 'timestamp', 'step', 'vector', 'item2vec', 'sim', 'reference', 'impressions',
                'rank')
        recom_click = recommand_cord.filter('reference==impressions')
        recom_click.cache()
        recom_click.withColumn('mrr', 1. / F.col('rank')).agg({"mrr": 'mean'}).show()













