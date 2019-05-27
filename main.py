import numpy as np
import pandas as pd

from model import EmbeddingHandler, LSTMArcFace
from utils import load_data, tokenize, add_validation_label
from os.path import exists

import torch
from torch import optim
from torch.autograd import Variable

def sampling(input_list, labels, batchsize):
    index = np.random.randint(0, len(input_list), batchsize)
    inputs = [input_list[i] for i in index]
    sort_index = np.argsort([-i.shape[0] for i in inputs])
    inputs = sorted(inputs, key=lambda x:x.shape[0], reverse=True)
    label = Variable(torch.LongTensor([labels[i] for i in index]))
    label = label[sort_index]
    
    return inputs, label



def train(model, optimizer, input_list, labels, valid_input, valid_labels, epoch=20, batchsize=16):

    for _ in range(epoch):
        Loss = 0
        for _ in range(100):
            optimizer.zero_grad()
            X, y = sampling(input_list, labels, batchsize)
            loss = model.get_loss(X, y)
            loss.backward()
            optimizer.step()
            X, y = sampling(valid_input, valid_labels, batchsize)
            model.eval()
            loss = model.get_loss(X, y)
            p = model(X, y)
            # print(np.mean((np.argmax(p.data.numpy(),axis=1) - y.data.numpy()) == 0))
            model.train()

            Loss += loss.data.cpu().numpy()

        print(Loss/100)
        torch.save(model.state_dict(), 'model.pth')



if __name__ == '__main__':
    df = load_data()
    if exists('./data_tokenized.csv') is not True:
        df = tokenize(df)
        df.to_csv('./data_tokenized.csv')
        df = pd.read_csv('./data_tokenized.csv')
    else:
        df = pd.read_csv('./data_tokenized.csv')

    df = add_validation_label(df)

    all_tokens = [str(t) for text_tokens in df['tokenized'] 
                    for tokens in eval(text_tokens)
                    for t in tokens]
    
    EH = EmbeddingHandler('./glove.6B.200d.txt', all_tokens)
    
    def _get_vectors(EH, index):
        li = []
        for i in index:
            texts = eval(df['tokenized'].iloc[i])
            matrix = EH.get_sentense_vectors(texts)
            li.append(Variable(torch.FloatTensor(matrix.T)))

        return li 

    index = np.where(df['validation']!=0)[0]
    input_list = _get_vectors(EH, index)
    labels = Variable(torch.LongTensor(df['label'].values[index]))

    index = np.where(df['validation']==0)[0]
    valid_input= _get_vectors(EH, index)
    valid_label= Variable(torch.LongTensor(df['label'].values[index]))

    model = LSTMArcFace(200, 100, len(df['label'].unique()))
    optimizer = optim.Adam(model.parameters())

    train(model, optimizer, input_list, labels, valid_input, valid_label)

    