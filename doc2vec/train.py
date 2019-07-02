from Resrec.doc2vec.model import  Doc2VecModel,SeqTrain
from Resrec.utils import  to_pickle,load_pickle
import pandas as pd
from Resrec.config import  train_path,test_path

if __name__=='__main__':
        train_data = pd.read_csv(train_path,nrows=3000)
        test_data = pd.read_csv(test_path, nrows=3000)
        data = pd.concat([train_data, test_data], axis=0)

        doc2vec_model = Doc2VecModel(vector_size=150, window=5, min_count=1, epochs=None, workers=5)
        doc2vec_model.fit_transform(data)
        doc2vec_model.transform_valid(test_data)

        to_pickle(doc2vec_model,'./models/doc2model.pkl')
        # docmodel = doc2vec_model.model
        # action_series = doc2vec_model.action_series
        # items_series = doc2vec_model.items_series

        doc2vec_model=load_pickle('./models/doc2model.pkl')
        train=SeqTrain(doc2vecmodel=doc2vec_model.model)
        train.transform(data)
        to_pickle(train,'./models/seq_train.pkl')