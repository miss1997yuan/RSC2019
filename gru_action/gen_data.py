#coding:utf-8
import numpy as np
import keras.preprocessing.sequence  as kps
import tensorflow as tf
from config import  *

def gen_data(item_raw=False):
        item_series = []  # 交互item 序列
        action_series = []  # 交互的action 序列
        impression_series = []  # 曝光序列
        click_item = []  # 曝光序列中被点击的item_id
        for _ in range(NUMS):
                n = np.random.randint(1, impression_length + 1)
                im = [np.random.randint(1, n_items) for _ in range(n)]
                impression_series.append(im)
                click_item.append([im[np.random.randint(0, n)]])
                n = np.random.randint(1, item_series_length + 1)
                item_series.append([np.random.randint(0, n_items) for _ in range(n)])
                action_series.append([np.random.randint(0, n_action) for _ in range(n)])

        labels = []
        for cl, im in zip(click_item, impression_series):
                iml = len(im)
                diffs = impression_length - iml
                pad = [im[np.random.randint(0, iml)] for _ in range(diffs)]
                im += pad
                labels.append((np.array(im) == cl[0]).astype(int))
        labels = np.array(labels)

        #padding
        action_series = kps.pad_sequences(action_series, item_series_length, padding='post')
        item_series = kps.pad_sequences(item_series, item_series_length, padding='post')
        impression_series = np.array(impression_series)
        if item_raw:
                item_info=gen_item_info()
                return item_series,action_series,impression_series,labels,item_info
        else:
                return item_series,action_series,impression_series,labels






def gen_item_info():
        item_info_data=np.zeros((n_items,item_info_dim))
        for i in range(n_items):
            n=np.random.randint(4,54)
            n_dims=[np.random.randint(0,55) for _ in range(n)]
            item_info_data[i,:][n_dims]=1.0
        return item_info_data


def getBatch(data_func):
        '''
        :param data_func:The  function of  generate data
        :return:  iteration
        '''
        dataset = tf.data.Dataset.from_tensor_slices((data_func()))
        dataset = dataset.repeat(epoches).batch(batch_size).shuffle(buffer_size=1000)
        iterator = dataset.make_initializable_iterator()
        return  iterator,iterator.get_next()

def getTestBatch(data_func):
        '''
        :param data_func:The  function of  generate data
        :return:  iteration
        '''
        dataset = tf.data.Dataset.from_tensor_slices((data_func()))
        dataset = dataset.repeat(epoches).batch(batch_size).shuffle(buffer_size=1000)
        iterator = dataset.make_initializable_iterator()
        return  iterator,iterator.get_next()


if __name__=='__main__':
#
        func_name=gen_data
        iterator,(item_series, action_series, impression_series, labels)= getBatch(func_name)
#         with tf.Session() as sess:
#                 sess.run(iterator.initializer)
#                 data=sess.run(item_series)
#                 print(data[0])







