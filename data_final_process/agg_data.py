from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import MapType, StringType, IntegerType
from collections import Counter

from pyspark.sql.types import FloatType, IntegerType, StructField, StructType, ArrayType

import numpy as np


@F.udf(ArrayType(StructType([
        # Adjust types to reflect data types
        StructField("item0", StringType()),
        StructField("item1", IntegerType()),
        StructField("item2", FloatType()),
        StructField("item3", IntegerType())

])))
def zip_imp_price(imp, price):
        imp_rank = range(len(imp))
        price = np.array(price).astype(float).tolist()
        from scipy.stats import rankdata
        price_rank = rankdata(price, method='dense').tolist()
        return zip(imp, imp_rank, price, price_rank)


@F.udf(returnType=MapType(StringType(), FloatType()))
def SessAction(act_map):
        actions = ['change of sort order',
                   'clickout item',
                   'filter selection',
                   'interaction item deals',
                   'interaction item image',
                   'interaction item info',
                   'interaction item rating',
                   'search for destination',
                   'search for item',
                   'search for poi']

        actions_map = dict(zip(actions, [0.0] * 10))

        actions_map.update(act_map)
        new_key = ['_'.join(line.split()) + "_sess" for line in actions_map.keys()]
        return dict(zip(new_key, actions_map.values()))


@F.udf(returnType=MapType(StringType(), IntegerType()))
def counter(r):
        if r:
                return dict(Counter(r.split('|')))
        else:
                return {}


zip_ = F.udf(
        lambda x, y: list(zip(x, y)),
        ArrayType(StructType([
                # Adjust types to reflect data types
                StructField("first", StringType()),
                StructField("second", StringType())
        ]))
)


@F.udf(ArrayType(FloatType()))
def sess_price(prices):
        prices = np.array(prices).astype('float')
        return float(prices.mean()), float(prices.max()), float(prices.std()), float(prices.min())


new_key = ['filter_selection_sess',
           'interaction_item_info_sess',
           'search_for_poi_sess',
           'clickout_item_sess',
           'interaction_item_rating_sess',
           'interaction_item_deals_sess',
           'change_of_sort_order_sess',
           'search_for_item_sess',
           'interaction_item_image_sess',
           'search_for_destination_sess']


@F.udf(returnType=MapType(StringType(), FloatType()))
def InteractionSessionMap(imp, ref_list, act_type):
        interaction_item = ['clickout item',
                            'interaction item deals',
                            'interaction item image',
                            'interaction item info',
                            'search for item',
                            'interaction item rating']

        inter_map = dict(zip(interaction_item, [0.0] * 6))
        if ref_list is None:
                pass
        else:
                if imp in ref_list:
                        mp = dict(Counter(np.array(act_type)[np.array(ref_list) == imp]))
                        inter_map.update(mp)
        k = ['_'.join(line.split()) + "_it" for line in inter_map.keys()]
        return dict(zip(k, inter_map.values()))


inter_select = ["*"] + ["tmp_map['%s'] as %s" % (line, line) for line in ['interaction_item_info_it',
                                                                          'clickout_item_it',
                                                                          'interaction_item_rating_it',
                                                                          'interaction_item_deals_it',
                                                                          'search_for_item_it',
                                                                          'interaction_item_image_it']]

if __name__ == '__main__':
        pth = " --master yarn    pyspark-shell"

        os.environ['PYSPARK_SUBMIT_ARGS'] = pth

        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.app.name', 'rec'), ('spark.executor.memory', '3g'),
                                   ('spark.executor.cores', '4'), ('spark.sql.shuffle.partitions', '300'),
                                   ('spark.cores.max', '6'), ('spark.executor.instances', '100'),
                                   ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()

        sqlContext.sql('use cmp_tmp')
        data = sqlContext.sql("select * from cmp_tmp_rec_train_data where reference is not NULL")

        # Step1:
        # 1.得出session 中clickout 最大的step 等同于session_size
        # 2.得出含有clickout 的session
        # 3.得出click 的时间
        agg_info = data.filter("action_type=='clickout item'").groupBy("session_id").agg(
                F.max('step').alias('click_step_max'),
                F.max('timestamp').alias('click_timestamp'))

        data = data.join(agg_info, on='session_id')

        # 得出click 的上下文信息
        data = data.filter("timestamp<=click_timestamp")
        data = data.withColumn('sess_rank',
                               F.rank().over(Window.partitionBy("session_id").orderBy("timestamp")).alias("sess_rank"))

        data = data.withColumn('time_diff', data.click_timestamp - data.timestamp)

        # click info
        click_data = data.filter('timestamp==click_timestamp')  # 865740
        # context info
        context_data = data.filter('timestamp<click_timestamp')

        # Step2:collect context info
        collect_cols = ['timestamp', 'step', 'action_type',
                        'reference', 'platform', 'city', 'device', 'current_filters', 'time_diff']
        funcs_collect = [F.concat_ws("|", F.collect_list(r)).alias(r + '_list') for r in collect_cols]

        # 聚合上下文信息
        context_data = context_data.groupBy('session_id').agg(*funcs_collect)
        # Step 3 连接上下文信息
        train_data = click_data.join(context_data, on='session_id', how='left')

        # Step 4
        # counter action type
        train_data = train_data.withColumn('action_type_map', counter('action_type_list'))
        # Step 5
        # explode data and get impression rank /impression and price rank /price
        train_data = train_data.withColumn('impressions', F.split("impressions", "\|")).withColumn(
                'prices', F.split("prices", "\|")).withColumn(
                "tmp", zip_imp_price("impressions", 'prices')).withColumn(
                "tmp", F.explode('tmp')).withColumn("impresssion", F.col("tmp.item0")).withColumn(
                'impress_rank', F.col("tmp.item1")).withColumn(
                "price", F.col("tmp.item2")).withColumn("price_rank", F.col("tmp.item3")).drop('tmp')

        # hour
        train_data = train_data.withColumn('hour', F.hour(F.from_unixtime(
                'timestamp', "yyyy/MM/dd HH:mm:ss")))

        # impression 价格特征
        train_data = train_data.withColumn("impress_len", F.size("impressions")).withColumn("tmp", sess_price(
                "prices")).withColumn(
                "sess_price_mean", F.col("tmp")[0]).withColumn("sess_price_max", F.col("tmp")[1]).withColumn(
                "sess_price_std", F.col("tmp")[2]).withColumn("sess_price_min", F.col("tmp")[3]).drop('tmp')

        # impression 的交互行为
        act_select = ["*"] + ["action_type_map['%s'] as %s" % (line, line) for line in new_key]
        train_data = train_data.withColumn("action_type_map", SessAction('action_type_map')).selectExpr(
                *act_select).drop("action_type_map")

        train_data = train_data.withColumn("reference_list", F.split("reference_list", "\|")
                                           ).withColumn("action_type_list",
                                                        F.split("action_type_list", "\|")).withColumn(
                'tmp_map', InteractionSessionMap("impression",
                                                 "reference_list", "action_type_list")).selectExpr(*inter_select)


        # 城市
        train_data = train_data.withColumn("country", F.split("city", "\,")[1]).withColumn("city",
                                                                                           F.split("city", "\,")[
                                                                                                   0]).select(
                "country", 'city')

        # Step 6 save
        train_data.write.saveAsTable("cmp_tmp_rec_train_agg", partitionBy=['flag', 'day'])
