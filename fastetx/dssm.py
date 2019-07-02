
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os

from pyspark.ml.feature import CountVectorizer,CountVectorizerModel
from pyspark.sql.types import ArrayType, StringType, IntegerType
import pyspark.sql.functions  as F
import numpy as np

model_pth='/team/cmp/hive_db/cmp_tmp/dl_model_template/Recsys/wyd/models'
model_name='dssm_cvmodel.model'

@F.udf(returnType=ArrayType(StringType()))
def item_seq(item_list, impression):
        app = ['0']
        for i in range(len(item_list) - 1):
                if item_list[i] != app[-1]:
                        app.append(item_list[i])
        app.append(impression)
        return app[1:]


digit_ref = F.udf(lambda arr: [line for line in arr if line.isdigit()], returnType=ArrayType(StringType()))
get_sparse_indices = F.udf(lambda sparse_vec: sparse_vec.indices.tolist(), returnType=ArrayType(IntegerType()))


def data_fit(data, cv_model):
        data = data.join(item_info, data.impression == item_info.item_id, how='inner')
        data = data.withColumn('item_list',
                               digit_ref('reference_list'))
        data = data.filter('size(item_list)>0'
                           ).withColumn('item_seq', item_seq('item_list', 'impression')).filter(
                "size(item_seq)<21 and size(item_seq)>0")
        data = data.withColumn("label", (data.reference == data.impression).cast('int'))

        data = data.drop('item_seq').withColumnRenamed("item_list", 'item_seq')

        data = cv_model.transform(data)
        data = data.withColumn('item_seq_enc', get_sparse_indices('item_seq_enc'))

        return data


def data_transform(data, cv_model):
        data = data_fit(data, cv_model)
        GR_COLS = ['user_id', 'session_id', 'timestamp', 'step', 'item_seq_enc', 'impression', 'label', 'flag']
        data = data.select(*GR_COLS)
        data = data.join(df_vocab, on='impression', how='inner')
        return data


if __name__=='__main__':
        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.executor.memory', '4g'),
                                   ('spark.executor.cores', '4'),('spark.driver.memory','10G'),
                                   ('spark.sql.shuffle.partitions', '1000'),
                                   ('spark.cores.max', '6'), ('spark.executor.instances', '200'),
                                   ('spark.network.timeout', '600s')])

        min_TF=20

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()

        sqlContext.sql("use cmp_tmp")
        # 对于历史曝光次数小与min_TF的进行过滤
        item_info = sqlContext.sql("select * from cmp_tmp_rec_item_info_feature ")
        item_info = item_info.filter("impression_freqs>%d"%min_TF)
        # reference_list is NULL 表示需要有上下文信息

        train_data = sqlContext.sql("select * from cmp_tmp_rec_train_agg where reference_list is not NULL ")
        test_data = sqlContext.sql(
                "select * from cmp_tmp_rec_test_agg where action_type=='clickout item'  and  reference_list is not  NULL")




        has_cv = True
        if has_cv:
                from pyspark.ml.feature import CountVectorizerModel

                cv = CountVectorizerModel()
                cv_model = cv.load(os.path.join(model_pth, model_name))

        else:
                cv = CountVectorizer(inputCol="item_seq", outputCol='item_seq_enc')
                cv_model = cv.fit(train_data)

        vocab = cv_model.vocabulary
        import pandas as pd

        vocab = pd.DataFrame(np.array([range(len(vocab)), vocab]).T, columns=['impression_id', 'impression'])
        df_vocab = sqlContext.createDataFrame(vocab)


        test_data=data_transform(test_data,cv_model)
        train_data=data_transform(train_data,cv_model)

