from pyspark.sql import SparkSession
from pyspark import SparkConf
import os

from pyspark.sql.types import *

import numpy as np
from collections import Counter
import pyspark.sql.functions as F
from pyspark.sql import Window


@F.udf(ArrayType(StructType([
        # Adjust types to reflect data types
        StructField("item0", StringType()),
        StructField("item1", IntegerType()),
        StructField("item2", FloatType()),
        StructField("item3", IntegerType())

])))
def zip_imp_price(imp, price):
        import numpy as np
        imp_rank = range(len(imp))
        price = np.array(price).astype(float).tolist()
        from scipy.stats import rankdata
        price_rank = rankdata(price, method='dense').tolist()
        return zip(imp, imp_rank, price, price_rank)


@F.udf(returnType=MapType(StringType(), IntegerType()))
def counter(r):
        if r:
                return dict(Counter(r.split('|')))
        else:
                return {}


@F.udf(ArrayType(FloatType()))
def sess_price(prices):
        prices = np.array(prices).astype('float')
        return float(prices.mean()), float(prices.max()), float(prices.std()), float(prices.min())


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
        return actions_map


@F.udf(returnType=MapType(StringType(), IntegerType()))
def InteractionSessionMap(imp, ref_list, act_type):
        import numpy as np
        if (ref_list != None) and (imp in ref_list):
                mp = dict(Counter(np.array(act_type)[np.array(ref_list) == imp].tolist()))



        else:
                mp = {}

        return mp


if __name__ == '__main__':
        pth = " --master yarn    pyspark-shell"

        os.environ['PYSPARK_SUBMIT_ARGS'] = pth

        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.app.name', 'rec'), ('spark.executor.memory', '4g'),
                                   ('spark.executor.cores', '4'), ('spark.sql.shuffle.partitions', '1000'),
                                   ('spark.cores.max', '6'), ('spark.executor.instances', '200'),
                                   ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()

        train = False

        if train:
                sqlContext.sql('use cmp_tmp')
                data = sqlContext.sql("select * from cmp_tmp_rec_train_data where reference is not NULL")

                # 1.得出session 中clickout 最大的step 等同于session_size
                # 2.得出含有clickout 的session
                # 3.得出click 的时间
                agg_info = data.filter("action_type=='clickout item'").groupBy("session_id").agg(
                        F.max('step').alias('click_step_max'),
                        F.max('timestamp').alias('click_timestamp'))

                data = data.join(agg_info, on='session_id')

                # 得出click 的上下文信息
                data = data.filter("timestamp<=click_timestamp")
                # 在session 中的顺序 针对存在两个step 的情况
                data = data.withColumn('sess_rank',
                                       F.rank().over(Window.partitionBy("session_id").orderBy("timestamp")).alias(
                                               "sess_rank"))

                data = data.withColumn('time_diff', data.click_timestamp - data.timestamp)

                click_data = data.filter("timestamp==click_timestamp and action_type=='clickout item'")  # 865740
                context_data = data.filter("timestamp<click_timestamp or action_type!='clickout item'")


        else:
                sqlContext.sql('use cmp_tmp')
                data = sqlContext.sql("select * from cmp_tmp_rec_train_data where flag='test'")
                agg_info = data.filter("action_type=='clickout item' and reference is NULL").selectExpr(
                        "session_id", "timestamp as click_timestamp  ", "step as click_step_max")
                data = data.join(agg_info, on='session_id')
                data = data.filter("timestamp<=click_timestamp")
                data = data.withColumn('sess_rank',
                                       F.rank().over(Window.partitionBy("session_id").orderBy("timestamp")).alias(
                                               "sess_rank"))
                data = data.withColumn('time_diff', data.click_timestamp - data.timestamp)
                click_cond = "action_type=='clickout item' and reference is NULL"
                click_data = data.filter("(%s)" % click_cond)  # 865740
                context_data = data.filter("not (%s)" % click_cond)

        collect_cols = ['timestamp', 'step', 'action_type',
                        'reference', 'platform', 'city', 'device', 'current_filters', 'time_diff']
        funcs_collect = [F.concat_ws("|", F.collect_list(r)).alias(r + '_list') for r in collect_cols]

        # 数据展开
        train_data = data.withColumn('impressions', F.split("impressions", "\|")).withColumn(
                'prices', F.split("prices", "\|")).withColumn(
                "tmp", zip_imp_price("impressions", 'prices')).withColumn(
                "tmp", F.explode('tmp')).withColumn("impression", F.col("tmp.item0")).withColumn(
                'impress_rank', F.col("tmp.item1")).withColumn(
                "price", F.col("tmp.item2")).withColumn("price_rank", F.col("tmp.item3")).drop('tmp')

        # hour
        train_data = train_data.withColumn('hour', F.hour(F.from_unixtime(
                'timestamp', "yyyy/MM/dd HH:mm:ss")))

        # country and city
        train_data = train_data.withColumn("country", F.split("city", "\,")[1]).withColumn("city",
                                                                                           F.split("city", "\,")[0])

        # impression 价格特征
        train_data = train_data.withColumn("impress_len", F.size("impressions")).withColumn("tmp", sess_price(
                "prices")).withColumn(
                "sess_price_mean", F.col("tmp")[0]).withColumn("sess_price_max", F.col("tmp")[1]).withColumn(
                "sess_price_std", F.col("tmp")[2]).withColumn("sess_price_min", F.col("tmp")[3]).drop('tmp')

        # session 内的action 统计

        # act_select=["*"]+["action_type_map['%s'] as %s" %(line,line)  for line in actions]
        # train_data=train_data.withColumn("action_type_map",SessAction('action_type_map')).selectExpr(*act_select).drop("action_type_map")

        train_data = train_data.withColumn("reference_list", F.split("reference_list", "\|")
                                           ).withColumn("action_type_list",
                                                        F.split("action_type_list", "\|")).withColumn(
                'interaction_map', InteractionSessionMap("impression",
                                                         "reference_list", "action_type_list"))

        train_data.write.saveAsTable("cmp_tmp_rec_test_agg", mode='overwrite', partitionBy=['flag', 'day'])


