from keras.layers   import Dense,Embedding,Input,GlobalAveragePooling1D
from keras  import Sequential,Model
import numpy as np
embed=(np.arange(0,20)/20.0).reshape(10,2)
model=Sequential()
model.add(Embedding(input_dim=10,output_dim=2,input_length=3,weights=[embed],trainable=False))
model.add(GlobalAveragePooling1D())
model.add(Dense(1,activation='sigmoid'))
X=model.predict(np.array([1,2,3]).reshape(1,3))
print(X)