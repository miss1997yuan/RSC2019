# coding:utf-8
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import FloatType
from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, StringType, BooleanType
import pyspark.sql.functions as F
import numpy as np
from operator import add
from functools import reduce


@F.udf(ArrayType(StructType([
        # Adjust types to reflect data types
        StructField("item0", StringType()),
        StructField("item1", IntegerType()),
        StructField("item2", FloatType())

])))
def ImpPrice(imp, price):
        imp_rank = range(len(imp))
        price = np.array(price).astype(float).tolist()
        return zip(imp, imp_rank, price)


def getPriceImpressionRank():
        funcs = []
        for col in ["price", 'imp_rank']:
                for func in [F.min, F.max, F.mean, F.stddev]:
                        funcs.append(func(col).alias(col + "_" + func.func_name))
        funcs.append(F.count("price").alias('impression_freqs'))
        return funcs


def is_digit(value):
        if value:
                return value.isdigit()
        else:
                return False


is_digit_udf = F.udf(is_digit, BooleanType())

if __name__ == '__main__':
        pth = " --master yarn    pyspark-shell"

        os.environ['PYSPARK_SUBMIT_ARGS'] = pth

        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.app.name', 'rec'), ('spark.executor.memory', '4g'),
                                   ('spark.executor.cores', '4'), ('spark.sql.shuffle.partitions', '1000'),
                                   ('spark.cores.max', '6'), ('spark.executor.instances', '200'),
                                   ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()

        item_info = sqlContext.sql("select * from cmp_tmp_rec_item_info")
        item_info = item_info.withColumn('properties', F.split('properties', '\|'))
        # step1 one hot raw item information
        cv = CountVectorizer(inputCol="properties", outputCol="vectors")
        model = cv.fit(item_info)
        item_info = model.transform(item_info)

        for i in range(157):
                name = 'item_raw%d' % i
                element = F.udf(lambda v: float(v[i]), FloatType())
                item_info = item_info.withColumn(name, element("vectors"))
        item_info = item_info.drop("properties").drop('vectors')



        # step2: history statistical information
        data = sqlContext.sql("select * from cmp_tmp_rec_train_data where  action_type='clickout item'")
        # feature1:prices
        item_price = data.select('impressions', 'prices').withColumn("impressions",
                                                                     F.split("impressions", '\|')).withColumn("prices",
                                                                                                              F.split(
                                                                                                                      "prices",
                                                                                                                      '\|')).withColumn(
                "tmp", ImpPrice("impressions", 'prices')).withColumn(
                "tmp", F.explode('tmp')).selectExpr(
                "tmp.item0 as item_id  ", 'tmp.item1  as imp_rank', 'tmp.item2  as price')

        funcs = getPriceImpressionRank()
        item_price = item_price.groupBy('item_id').agg(*funcs)
        # feature 2  interaction item information For example  the freqs of user interact with item image
        data = sqlContext.sql("select * from cmp_tmp_rec_train_data where  reference is not NULL")
        item_action = data.filter(is_digit_udf("reference")).cube("reference", 'action_type').count()

        interaction_item = ['clickout item',
                            'interaction item deals',
                            'interaction item image',
                            'interaction item info',
                            'search for item',
                            'interaction item rating']

        item_action = item_action.groupby("reference").pivot('action_type', interaction_item).sum('count').fillna(0)
        # freqs
        for col in interaction_item:
                col_ = '_'.join(col.split()) + '_freq_con'
                item_action = item_action.withColumnRenamed(col, col_)

        item_action = item_action.withColumnRenamed("reference", 'item_id')

        numeric_col_list = ['clickout_item_freq_con',
                            'interaction_item_deals_freq_con',
                            'interaction_item_image_freq_con',
                            'interaction_item_info_freq_con',
                            'search_for_item_freq_con',
                            'interaction_item_rating_freq_con']

        item_action = item_action.withColumn('interaction_sum', reduce(add, [F.col(x) for x in numeric_col_list]))



        # the ratio of interaction types
        for col in numeric_col_list:
                item_action = item_action.withColumn(col + '_ratio', item_action[col] / item_action['interaction_sum'])
        # step3：Aggregate data
        item_info = item_info.join(item_price, on='item_id', how='left')
        item_info = item_info.join(item_action, on='item_id', how='left')





