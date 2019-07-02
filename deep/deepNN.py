import pandas as pd

from keras.layers import Dense
from keras import Sequential

train_path='data_process/train_set.csv'
test_path='data_process/test_data.csv'
valid_path='dataset/valid_set.csv'
def transform_data(train_data, test_data):
        train_data.drop(['item_id'], axis=1, inplace=True)
        test_data.drop(['item_id'], axis=1, inplace=True)
        Y = train_data.label
        X = train_data.drop('label', axis=1)
        Y_valid = test_data.label
        X_valid = test_data.drop('label', axis=1)

        return X, Y, X_valid, Y_valid


if __name__=='__main__':

        train_data=pd.read_csv(train_path,nrows=2000)
        test_data=pd.read_csv(test_path,nrows=3000)

        X,Y,X_valid,Y_valid=transform_data(train_data,test_data)

        from scipy import sparse

        from sklearn.preprocessing  import StandardScaler
        input_dim=X.shape[1]

        standard=StandardScaler()
        standard=standard.fit(X)


        X=sparse.csr_matrix(standard.fit_transform(X))
        X_valid=sparse.csr_matrix(standard.fit_transform(X_valid))


        model=Sequential()
        model.add(Dense(100,input_shape=[input_dim,],activation='relu'))

        model.add(Dense(10,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(optimizer='adam',loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(X,Y,epochs=10, batch_size=20,shuffle=True)

        model.save("./models/deep.h5")
