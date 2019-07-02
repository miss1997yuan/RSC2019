import tensorflow as tf
from utils import  mrr,length
from config import  *
import keras
from gen_data import *
import numpy as np

with tf.name_scope("Input"):
        Input_Item=tf.placeholder(dtype=tf.int32,shape=[None,item_series_length],name='item_input')
        Input_action=tf.placeholder(dtype=tf.int32,shape=[None,item_series_length],name='action_input')
        Input_Impression=tf.placeholder(dtype=tf.int32,shape=[None,impression_length],name='impression_input')
        Input_label=tf.placeholder(tf.int32,shape=[None,15],name="label_input")

with tf.name_scope("Embedding"):
        # initializer = tf.random_normal_initializer(mean=0, stddev=1.0)
        initializer=tf.glorot_normal_initializer()
        embedding_item=tf.get_variable('embedding_item',[n_items,embedding_size],initializer=initializer)
        embedding_action=tf.get_variable('embedding_action',[n_action,1],initializer=initializer)

        softmax_W = tf.get_variable('softmax_w', [n_items, num_hidden], initializer=initializer)
        softmax_b = tf.get_variable('softmax_b', [n_items], initializer=tf.constant_initializer(0.0))

with tf.name_scope("MutiplyBatchNorm"):
        Item_embeded=tf.nn.embedding_lookup(embedding_item,Input_Item)
        action_embeded=tf.nn.embedding_lookup(embedding_action,Input_action)

        Item_embeded=tf.multiply(Item_embeded,action_embeded)
        #bathc normalization
        Item_embeded=keras.layers.BatchNormalization()(Item_embeded)

with tf.name_scope("dynamicGRU"):
        cell= tf.nn.rnn_cell.GRUCell(num_hidden,activation=tf.nn.relu)
        drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=dropout_p_hidden)
        len_seq=length(tf.expand_dims(Input_Item,2))

        output, state = tf.nn.dynamic_rnn(
            drop_cell,
            Item_embeded,
            dtype=tf.float32,
            sequence_length=len_seq,
        )
        with tf.name_scope("LastOutput"):
                #获取最后的输出
                batch_range = tf.range(tf.shape(output)[0])
                indices = tf.stack([batch_range, len_seq-1], axis=1)
                last = tf.gather_nd(output, indices)

with tf.name_scope("NegativeSampling"):
        sampled_W=tf.nn.embedding_lookup(softmax_W,Input_Impression)
        # sampled_b = tf.nn.embedding_lookup(softmax_b,Input_Impression)
        logits=tf.squeeze(tf.matmul(tf.expand_dims(last,1), tf.transpose(sampled_W,[0,2,1])),1)

with tf.name_scope("prediction"):
        prediction=tf.nn.softmax(logits,name='prediction')  #预测值
        MRR_op = mrr(logits, Input_label)

with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Input_label))
        train_op=tf.train.AdamOptimizer().minimize(cost)

# summary
tf.summary.scalar('loss',cost)
tf.summary.scalar('MRR',MRR_op)
merged_summary_op=tf.summary.merge_all()

# saver
saver=tf.train.Saver(max_to_keep=5)

func_name=gen_data
iterator,Batch= getBatch(func_name)
init_op = tf.global_variables_initializer()



with tf.Session() as sess:
        sess.run([iterator.initializer,init_op])

        summary_writer=tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())


        mean_loss , mean_mrr =np.zeros(STEP),np.zeros(STEP)

        for step in range(20000):
                try:
                        batch_data = sess.run(Batch)
                        feed_dict = {Input_Item: batch_data[0],
                                             Input_action: batch_data[1],
                                             Input_Impression: batch_data[2].reshape(-1, 15),
                                             Input_label: batch_data[3]}
                        idx=step%STEP
                        mean_loss[idx], mean_mrr[idx], summary = sess.run([cost, MRR_op, merged_summary_op], feed_dict)
                        summary_writer.add_summary(summary, step)

                        if step%STEP==0:
                                print(f'Step{step} Loss : {mean_loss.mean()}  MRR:{mean_mrr.mean()}')
                                mean_loss, mean_mrr = np.zeros(STEP), np.zeros(STEP)
                        if step%100==0:
                                # saver.save(sess,f'{logs_path}/{MODEL_NAME}')
                                saver.save(sess,f'{logs_path}/{MODEL_NAME}',global_step=step,write_meta_graph=False)

                except tf.errors.OutOfRangeError:
                        print("模型训练结束")
                        break







