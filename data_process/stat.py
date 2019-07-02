
from config import  test_path
import pandas as pd
from utils import  addflag
from multiprocessing import Pool
def step_max(sess):
    return  sess.step.max(),sess.query("check_ref==1 and  action_type=='clickout item'").step.max()


def apply_stat(chunk):
        chunk = addflag(chunk)
        chunk = chunk.groupby('session_id').apply(step_max)
        chunk.to_csv('stat_ste.csv',mode='a+')


if __name__=='__main__':
        chunk = pd.read_csv(test_path, nrows=100)



        test_data=pd.read_csv(test_path,chunksize=200000)
        p=Pool(5)
        for chunk in test_data:
                p.apply_async(apply_stat,args=(chunk,))
        p.close()
        p.join()


