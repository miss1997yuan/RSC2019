import   tensorflow as tf
import  numpy as np

n_items = 50
adj_size = 10
epoches=100
n_dim=8
nn=np.arange(50)
samples=np.array([nn[i:i+10] for i in range(len(nn)-10)])
samples=np.tile(samples,epoches).reshape(-1,10)
np.random.shuffle(samples)
labels=np.array([(np.argsort(line)==4).astype('int')  for line  in  samples])
Y = np.array([line[np.argsort(line) == 4] for line in samples])


indx=tf.placeholder(tf.int32,shape=[None,adj_size])
Y_input=tf.placeholder(tf.int32,shape=[None,adj_size])

embed=tf.Variable(tf.random_normal(shape=[n_items,n_dim]))
embedding=tf.nn.embedding_lookup(embed,indx)
embedding=tf.reduce_mean(embedding,axis=1)

init_op=tf.global_variables_initializer()

with tf.Session()  as sess:
        sess.run(init_op)
        zz=sess.run(embedding,feed_dict={indx:np.array([1,2,4,5,6,7,8,9,10,11]).reshape(-1,10)})
        print(zz.shape)

