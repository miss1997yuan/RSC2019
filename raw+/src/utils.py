import gensim
import numpy as np
import pandas as pd


def context_sess(line):
        sereis = []
        tmp = []
        for r in line:
                if isinstance(r, str):
                        if r.isdigit():
                                tmp.append(r)
                        else:
                                if len(tmp) > 1:
                                        sereis.append(tmp)
                                tmp = []
        if len(tmp) > 1:
                sereis.append(tmp)
        return sereis


def context_items(train, test):
        sents = []
        sess_list = train.groupby("session_id").apply(lambda r: r['reference'].tolist())
        sess_list.apply(context_sess).apply(sents.extend)

        test.groupby('session_id').apply(lambda r: r['reference'
        ].tolist()).apply(context_sess).apply(sents.extend)

        return sents


def pretrain_item2vec(sents, dim=20):
        model = gensim.models.Word2Vec(sents, size=dim)
        item2vec = model.wv.vectors
        # mask 0
        item2vec = np.vstack((np.zeros((1, 20)) + 0.001, item2vec))

        index2word = model.wv.index2word
        # item from
        index2word = dict(zip(index2word, range(1, len(index2word) + 1)))

        print("item size:", item2vec.shape[0])
        return item2vec, index2word


def generate_data(df, index2word, mode='train'):
        item2id = lambda line: [index2word.get(r) for r in line if r in index2word]
        sess_list = df.groupby("session_id").apply(lambda r: r['reference'].tolist()).apply(item2id)
        if mode == 'train':
                df = df[df.groupby("session_id").cumcount(ascending=False) == 0][df.action_type == 'clickout item']
        else:
                df = df[df.reference.isna()][df.action_type == 'clickout item']

        df = df.loc[:, ['session_id', 'reference', 'impressions']]

        df.impressions = df.impressions.str.split("|")
        df['impressions2id'] = df.impressions.apply(lambda r: [index2word.get(w, 0) for w in r if w in index2word])
        df['reference'] = df.reference.apply(lambda r: index2word.get(r, 0))

        df = pd.merge(df, sess_list.reset_index(), on='session_id', how='left')

        df.rename(columns={0: 'context'}, inplace=True)
        df.context = df.context.apply(lambda r: r[:-1])

        from keras.preprocessing.sequence import pad_sequences
        target, impression, context = df.reference.values.reshape(-1, 1) \
                , pad_sequences(df.impressions2id.values, maxlen=25), \
                                      pad_sequences(df.context.values, maxlen=30)

        return target, impression, context
