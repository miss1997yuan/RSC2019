from keras.layers import Dense, GlobalAvgPool1D,Embedding
from keras import Sequential
import numpy  as np


# nums = np.arange(1, 101)
# n_samples = 1000

# samples = np.array([np.random.randint(0, n_items, adj_size) for i in range(n_samples)])
# labels = np.array([(np.argsort(line) == 4).astype('int') for line in samples])

n_items = 50
adj_size = 10
epoches=100
nn=np.arange(50)
samples=np.array([nn[i:i+10] for i in range(len(nn)-10)])
samples=np.tile(samples,epoches).reshape(-1,10)
np.random.shuffle(samples)
labels=np.array([(np.argsort(line)==4).astype('int')  for line  in  samples])
Y = np.array([line[np.argsort(line) == 4] for line in samples])


model=Sequential()
model.add(Embedding(input_dim=n_items,output_dim=8,input_length=adj_size))
# print(samples[0])
model.add(GlobalAvgPool1D())
model.add(Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(samples,labels,epochs=1000,batch_size=50,validation_split=0.3)
# cc=model.predict(np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,10))
# print(cc)