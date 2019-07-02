from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import Window
import pyspark.sql.functions as F
import os
import argparse
from pyspark.sql.types import ArrayType, StringType

model_pth = '/team/cmp/hive_db/cmp_tmp/dl_model_template/Recsys/wyd/models'
model_name = 'item2vec_len15.model'

GR_COLS = ["user_id", "session_id", "timestamp", "step"]
all_cols = GR_COLS + ['impression', 'score']


@F.udf
def getCosinDis(item, session):
        from scipy.spatial import distance
        if (item == None) | (session == None):
                return 5.
        else:
                return round(float(distance.cosine(item.toArray(), session.toArray())), 5)


def Doc2vec(args, train_data, vector_size=10, window_size=5, input_col='reference_list'):
        word2Vec = Word2Vec(vectorSize=vector_size, minCount=2, seed=42, maxIter=1, windowSize=window_size,
                            inputCol=input_col,
                            outputCol="ref_vec")
        model = word2Vec.fit(train_data)
        model_name = "{}_{}_{}_{}".format(args.model_name, vector_size, window_size, args.is_item)
        model.write().overwrite().save(os.path.join(model_pth, model_name))
        vec = model.getVectors()

        train_data = model.transform(train_data)
        train_data = train_data.join(vec, train_data.impression == vec.word, how='left')
        train_data = train_data.withColumn('score', getCosinDis("ref_vec", 'vector'))
        train_data = train_data.withColumn("row_number",
                                           F.rank().over(
                                                   Window.partitionBy(GR_COLS).orderBy(train_data["score"].desc())))

        data = train_data.select(GR_COLS + ['action_type', 'impressions', 'reference',
                                            'score', 'row_number', 'impression']).filter("reference==impression")

        demo_cnt = data.cube("row_number").count().toPandas()
        demo_cnt = demo_cnt.dropna()
        # demo_cnt.toPandas(".csv".format(model_name))
        MRR = ((demo_cnt['count'] / demo_cnt['count'].sum()) * (1. / (26.0 - demo_cnt['row_number']))).sum()
        print('**---**' * 20)
        print("InputCol:{} vector_size:{} window_size:{} MRR:{}".format(input_col, vector_size, window_size, MRR))

        return MRR


if __name__ == '__main__':
        # nohup spark-submit   --master yarn --name doc_vector_size_test_item   doc2_test1.py   --model_name doc2vec_win  --is_item 1 > item_list.log 2>&1 &

        parse = argparse.ArgumentParser()
        # parse.add_argument('--size',type=int,default=10,help='vector size')
        parse.add_argument('--model_name', type=str, default='doc2vec_len2')

        parse.add_argument('--is_item', type=int, default=0, help='item2vev if 1 else ref2vec')

        args = parse.parse_args()

        conf = SparkConf().setAll([('spark.master', 'yarn'), ('spark.executor.memory', '4g'),
                                   ('spark.executor.cores', '4'), ('spark.driver.memory', '10G'),
                                   ('spark.sql.shuffle.partitions', '1000'),
                                   ('spark.cores.max', '6'), ('spark.executor.instances', '200'),
                                   ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()
        sqlContext.sql("use cmp_tmp")
        train_data = sqlContext.sql("select * from cmp_tmp_rec_train_agg where reference_list is not NULL ")

        digit_ref = F.udf(lambda arr: [line for line in arr if line.isdigit()], returnType=ArrayType(StringType()))

        train_data = train_data.withColumn('item_list',
                                           digit_ref('reference_list')).filter('size(item_list)>1')

        train_data.cache()
        if args.is_item == 0:
                input_col = 'reference_list'
        else:
                input_col = 'item_list'

        MRR_size = []
        vec_sizes = [8, 15, 20, 50, 100, 150, 200, 300]
        print("test doc2vec size ")
        max_mrr = 0.0
        best_vector_size = 5
        for size in vec_sizes:
                MRR = Doc2vec(args, train_data, vector_size=size, input_col=input_col)
                if max_mrr < MRR:
                        best_vector_size = size
                MRR_size.append(MRR)

        wMRRs = []
        window_sizes = [5, 6, 7, 8, 9, 10]
        for size in window_sizes:
                MRR = Doc2vec(args, train_data, vector_size=best_vector_size, window_size=size, input_col=input_col)
                wMRRs.append(MRR)

        import pandas as pd

        vec_data = pd.Series(index=vec_sizes, data=MRR_size)
        vec_data.index.name = 'vector_size'
        vec_data.to_csv("%s_vector_size.csv" % input_col)

        vec_data = pd.Series(index=window_sizes, data=wMRRs)
        vec_data.index.name = 'window_size'
        vec_data.to_csv("%s_window_size.csv" % input_col)








