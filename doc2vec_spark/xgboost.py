from pyspark.sql import SparkSession
from pyspark import SparkConf
import os
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import Window

from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature  import StandardScaler
from pyspark.sql.types import DoubleType
from pyspark.sql import Window
import numpy as np
interaction_item = ['clickout item',
                    'interaction item deals',
                    'interaction item image',
                    'interaction item info',
                    'search for item',
                    'interaction item rating']

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


def map2row(df):
        concat_ws = lambda line, suf: '_'.join(line.split()) + "_" + suf
        for act_ in actions:
                map_udf = F.udf(lambda map_: map_.get(act_, 0.0), returnType=FloatType())
                df = df.withColumn(concat_ws(act_, '_sess'), map_udf("action_type_map"))

        for inter_ in interaction_item:
                map_udf = F.udf(lambda map_: map_.get(inter_, 0.0), returnType=FloatType())
                df = df.withColumn(concat_ws(inter_, '_it'), map_udf("interaction_map"))

        df = df.drop('action_type_map').drop('interaction_map').fillna(0.0)
        return df


train = pd.read_csv('data_process/train_set.csv', nrows=2).drop(['item_id', 'label'], axis=1)
use_cols = train.columns.to_list()


@F.udf(returnType=FloatType())
def getProb(*cols):
        # 每个叶子节点的值*树的权重在运用一下公式
        import math
        sum_ = reduce(lambda x, y: x + y, cols)
        prob = 1.0 / (1.0 + math.exp(-2 * sum_))
        return prob


def gbt_transform(gbt_model, df_vec):
        trees = gbt_model.trees
        trees_weights = gbt_model.treeWeights
        for i, (bt, bt_w) in enumerate(zip(trees, trees_weights)):
                df_vec = bt.transform(df_vec)
                df_vec = df_vec.withColumn('pre%dtree' % i, bt_w * df_vec['prediction'])
                df_vec = df_vec.drop("prediction")

        columns = df_vec.columns
        prob_col = [line for line in columns if line.endswith('tree')]
        df_vec = df_vec.withColumn('gbt_probability', getProb(*prob_col))
        for c in prob_col:
                df_vec = df_vec.drop(c)
        df_vec = gbd_model.transform(df_vec)
        return df_vec


def evaluator(df):
        biclass = BinaryClassificationEvaluator()
        bieval = biclass.evaluate(df)
        predictionAndLabels = df.rdd.map(lambda row: (float(row['prediction']), float(row['label'])))
        metrics = MulticlassMetrics(predictionAndLabels)
        confusion_matrix = metrics.confusionMatrix().toArray()
        return bieval, confusion_matrix



if __name__=='__main__':


        pth = " --master yarn    pyspark-shell"

        os.environ['PYSPARK_SUBMIT_ARGS'] = pth

        conf = SparkConf().setAll(
                [('spark.master', 'yarn'), ('spark.app.name', 'rec_model'), ('spark.executor.memory', '4g'),
                 ('spark.executor.cores', '4'), ('spark.driver.memory', '10G'),
                 ('spark.sql.shuffle.partitions', '1000'),
                 ('spark.cores.max', '6'), ('spark.executor.instances', '200'),
                 ('spark.network.timeout', '600s')])

        sqlContext = SparkSession.builder.master('yarn').config(conf=conf).enableHiveSupport().getOrCreate()
        # step1：数据转换
        train_set = sqlContext.sql("select * from  cmp_tmp_rec_train_set ")
        test_set = sqlContext.sql("select * from  cmp_tmp_rec_test_set")

        train_set = map2row(train_set)
        test_set = map2row(test_set)
        # step2: 验证集合
        valid_set = sqlContext.sql("select * from  cmp_tmp_rec_valid_set")
        valid_set = valid_set.withColumn('label', F.lit(1))
        valid_set = map2row(valid_set)
        # step3: 模型
        vec = VectorAssembler(inputCols=use_cols, outputCol='feature_vec')
        stander = StandardScaler(inputCol='feature_vec', outputCol="features")
        gbt = GBTClassifier(maxIter=50)
        pipline = Pipeline(stages=[vec, stander, gbt])
        model = pipline.fit(train_set)
        train_set = model.transform(train_set)
        test_set = model.transform(test_set)
        train_auc, train_confusion = evaluator(train_set)
        test_auc, test_confusion = evaluator(test_set)


        # step5 提交数据
        valid_set = model.transform(valid_set)
        get1 = F.udf(lambda v: float(np.float64(v[1])), returnType=FloatType())

        valid_set = valid_set.withColumn("score", get1('probability'))
        GR_COLS = ["user_id", "session_id", "timestamp", "step"]
        all_cols = GR_COLS + ['impression', 'score']

        submission_data = valid_set.select(*all_cols)



        submission_data = submission_data.withColumn("row_number",
                                                     F.rank().over(Window.partitionBy(GR_COLS).orderBy(
                                                             submission_data["score"].desc())))

        sub_data_list = submission_data.groupby(GR_COLS).agg(F.collect_list("impression").alias("item_recommendations"))
        sub_data_list = sub_data_list.withColumn("item_recommendations", F.concat_ws(" ", 'item_recommendations'))
        sub_data_list.toPandas().to_csv(pth, index=False)
