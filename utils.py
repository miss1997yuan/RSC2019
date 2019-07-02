import tensorflow as tf

import pickle
import  matplotlib.pyplot as plt
import  logging
def mrr(logits, labels):
        rank = tf.arg_max(
                tf.cast(
                        tf.equal(
                                tf.argsort(
                                        logits, 1), tf.expand_dims(tf.cast(tf.argmax(labels, 1), tf.int32), 1)),
                        tf.int32), dimension=1)
        return tf.reduce_mean(tf.divide(1, tf.add(rank, 1)))


def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length


def inted_reference(series):
        try:
                int(series['reference'])
                return 1
        except:
                return 0


def to_pickle(obj, file_name):
        with open(file_name, 'wb') as f:
                pickle.dump(obj, f)


def load_pickle(file_name):
        with open(file_name, 'rb') as f:
                return pickle.load(f)


def plot_distribution(row_count, title=None, name="item feature "):
        row_count_ratio = row_count / row_count.sum()
        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 13))
        row_count_ratio[:40].plot.bar(ax=ax0, rot=60, title='The number{}'.format(name))

        row_count_ratio.cumsum()[:40].plot.bar(ax=ax1, rot=60, title='The cumsum ratio{}'.format(name))


def getLogger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger=getLogger()
logger_info=lambda info:logger.info(info)



# if __name__=='__main__':
#         logs_path
