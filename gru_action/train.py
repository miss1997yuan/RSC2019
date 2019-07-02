
from models import  GRU4REC
from preprocess import Paded,GRUExtractor
from utils  import to_pickle,load_pickle
import tensorflow as tf
import argparse,os

def parse_args():
        parser = argparse.ArgumentParser(description='GRU4Rec args')

        parser.add_argument('--size', default=100, type=int, help='item embedding size ')
        parser.add_argument('--epoch', default=3, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--train', default=1, type=int)
        parser.add_argument('--num_hidden', default=120, type=int)
        parser.add_argument('--dropout', default=0.6, type=float)
        parser.add_argument('--logdir', default='./checkpoint', type=str)
        return  parser.parse_args()



def set_config():
        os.environ['CUDA_VISIBLE_DEVICES']='1'

        gpu_options=tf.GPUOptions(per_process_gpu_mermory_fraction=0.8)

        return  tf.ConfigProto(gpu_options=gpu_options)






if __name__=='__main__':
        args = parse_args()

        data = load_pickle('./processed_data/gru_data_shift_30.pkl')
        pad = Paded()
        iterator, Batch = pad.fit_transform(data, epoches=100, batch_size=20)
        pad.checkpoint_dir = args.logdir
        pad.embedding_size = args.size
        pad.dropout_p_hidden=args.dropout
        pad.num_hidden = args.num_hidden
        #
        sess = tf.Session()
        model= GRU4REC(sess=sess, args=pad)
        model.fit(iterator,Batch,verbose=20)
