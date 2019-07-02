
from networkx import karate_club_graph,to_numpy_matrix
import numpy as np


class karate(object):

        zkc=karate_club_graph()
        order=sorted(list(zkc.nodes()))

        NODE_SIZE=len(order)
        #Adjacency matrix
        A=to_numpy_matrix(zkc,nodelist=order)

        #Unit matrix college
        I=np.eye(zkc.number_of_nodes())

        node_label = []
        for i in range(34):
                label = zkc.node[i]
                if label['club'] == 'Officer':
                        node_label.append(1)
                else:
                        node_label.append(0)




