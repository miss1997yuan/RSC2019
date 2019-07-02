import os
path='../../datasets'
train_path = os.path.join(path,'train.csv')
test_path = os.path.join(path,'test.csv')
item_path = os.path.join(path,'item_metadata.csv')


actions=['change of sort order',
 'clickout item',
 'filter selection',
 'interaction item deals',
 'interaction item image',
 'interaction item info',
 'interaction item rating',
 'search for destination',
 'search for item',
 'search for poi']



interaction_item=['clickout item',
 'interaction item deals',
 'interaction item image',
 'interaction item info',
 'search for item',
 'interaction item rating']