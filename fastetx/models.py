import tensorflow as tf
from keras.layers    import Dense,Embedding
import numpy as np

embed=(np.arange(0,20)/20.0).reshape(10,2)

print(embed[0])

indx=tf.placeholder(tf.int32,shape=[None,None])
embedding=tf.Variable(initial_value=embed,trainable=False)
embedding_lookup=tf.nn.embedding_lookup(embedding,indx)

doc2vec=tf.reduce_mean(embedding,axis=0)

init_op=tf.global_variables_initializer()


with tf.Session()  as sess:
        sess.run(init_op)
        feed_dict={indx:[[1,2,3,4,4],[2,3,0,0,0]]}
        print(sess.run(doc2vec,feed_dict))





