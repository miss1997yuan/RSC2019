import sys
sys.path.append('..')
from config import  test_path
from utils import *

from  multiprocessing   import Pool


def apply_transform(sess):
        df=get_sess_context(sess)
        df.to_csv('test_data.csv',index=False,header=False,mode='a+')


if __name__=='__main__':

        with open('test_data.csv','w') as f:
                f.write(test_headder)
                f.write('\n')

        test_data = pd.read_csv(test_path, nrows=100)
        test_data = addflag(test_data)
        group_data=test_data.groupby("day_month")


        p=Pool(3)
        for _,sess in group_data:
                p.apply_async(apply_transform,args=(sess,))
        p.close()
        p.join()






