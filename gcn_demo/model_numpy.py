from networkx import karate_club_graph,to_numpy_matrix
import numpy as np
import pandas as pd
zkc=karate_club_graph()
order=sorted(list(zkc.nodes()))

NODE_SIZE=len(order)
#Adjacency matrix
A=to_numpy_matrix(zkc,nodelist=order)

#Unit matrix college
I=np.eye(zkc.number_of_nodes())


# self loops
A_hat=A+I

#Standardized  adjacency matrix
D_hat=np.array(np.sum(A_hat,axis=0))[0]
D_hat=np.matrix(np.diag(D_hat))


W1=np.random.normal(loc=0,scale=1,size=(zkc.number_of_nodes(),4))
W2=np.random.normal(loc=1,size=(W1.shape[1],2))


relu=lambda x:np.maximum(0,x)

def gcn_layer(A_hat,D_hat,X,W):
    return relu(D_hat**-1*A_hat*X*W)


H1=gcn_layer(A_hat,D_hat,I,W1)
H2=gcn_layer(A_hat,D_hat,H1,W2)

output=H2


feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}


node_label=[]
for i in range(NODE_SIZE):
    label=zkc.node[i]
    if label['club']=='Officer':
        node_label.append(1)
    else:
        node_label.append(0)

import matplotlib.pyplot as plt
represent=pd.DataFrame(feature_representations).T.values
plt.scatter(represent[:,0],represent[:,1],s=200,c=node_label)
plt.show()