import tensorflow as tf
from gen_data import  getTestBatch,gen_data
print(tf.__version__)

saver=tf.train.import_meta_graph('./checkpoint/gru-model.meta')
with tf.Session() as sess:
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]


        saver.restore(sess,'./checkpoint/gru-model-1400')
        graph = tf.get_default_graph()
        Input_Item =graph.get_tensor_by_name("Input/item_input:0")
        Input_action =graph.get_tensor_by_name("Input/action_input:0")
        Input_Impression =graph.get_tensor_by_name("Input/impression_input:0")
        Input_label =graph.get_tensor_by_name("Input/label_input:0")
        prediction =graph.get_tensor_by_name("prediction/prediction:0")
        func_name=gen_data
        iterator, Batch = getTestBatch(func_name)
        sess.run(iterator.initializer)
        s=tf.argsort(prediction)

        for line in range(10):
                batch_data=sess.run(Batch)
                feed_dict = {Input_Item: batch_data[0],
                             Input_action: batch_data[1],
                             Input_Impression: batch_data[2].reshape(-1, 15),
                             Input_label: batch_data[3]}
                p=sess.run(s,feed_dict)
        print(batch_data[0].tolist(),p)





