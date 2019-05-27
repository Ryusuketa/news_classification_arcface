import numpy as np
import pandas as pd
from model import EmbeddingHandler

from utils import load_data, tokenize, add_validation_label

from os.path import exists

from sklearn.ensemble import GradientBoostingClassifier
import pickle


if __name__ == '__main__':
    df = load_data()
    if exists('./data_tokenized.csv') is not True:
        df = tokenize(df)
        df.to_csv('./data_tokenized.csv')
        df = pd.read_csv('./data_tokenized.csv')
    else:
        df = pd.read_csv('./data_tokenized.csv')

    all_tokens = [str(t) for text_tokens in df['tokenized'] 
                    for tokens in eval(text_tokens)
                    for t in tokens]
    EH = EmbeddingHandler('./glove.6B.200d.txt', all_tokens)
    
    business_idx = df[df['type'] == 'business'].index[:200]
    tech_idx = df[df['type'] == 'politics'].index[:200]
    entertainment_idx = df[df['type'] == 'entertainment'].index[:200]

    def _get_vectors(EH, index):
        li = []
        for i in index:
            texts = eval(df['tokenized'].iloc[i])
            matrix = EH.get_sentense_vectors(texts)
            vector = np.mean(matrix, axis=1)
            #vector = vector/np.linalg.norm(vector)
            li.append(vector)

        return li

    business = np.array(_get_vectors(EH, business_idx))
    tech = np.array(_get_vectors(EH, tech_idx))
    entertainment = np.array(_get_vectors(EH, entertainment_idx))

    train = np.concatenate([business[:90],tech[:90], entertainment[:90]])
    test = np.concatenate([business[90:],tech[90:], entertainment[90:]])
    # labels = np.concatenate([np.ones(90)*i for i in range(3)])
    labels = np.concatenate([np.ones(90)*0,np.ones(180)*1])

    model = GradientBoostingClassifier()
    model.fit(train,labels)

    labels = np.concatenate([np.ones(110)*0,np.ones(220)*1])
    p = model.predict(test)
    prec = np.sum(p[:110]==0)/np.sum(p==0)
    recl = np.sum(p[:110]==0)/110
    print(prec, recl)
    
    
