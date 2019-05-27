import csv
csv.field_size_limit(100000000000)
import pandas as pd
import numpy as np
from scipy.linalg import svd

from typing import List 

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence 
from torch.nn.utils.rnn import pad_packed_sequence 
from torch.nn.utils.rnn import pack_padded_sequence 

from arcface_pytorch.models.metrics import ArcMarginProduct

class EmbeddingHandler(object):
    def __init__(self, glove_vector_path, all_tokens: List[str], alpha = 1e-4):
        self.vectors, self.token2id = self.get_glove_vectors(glove_vector_path)
        self.known_tokens = self.token2id.keys()
        self.word_prob = self._get_token_probability(all_tokens)
        self.alpha = alpha

    def _get_token_probability(self, all_tokens):
        S = pd.Series(all_tokens)
        unique_tokens = S.unique()
        counts = S.value_counts()
        word_prob = {t: counts[t]/len(S) for t in unique_tokens}

        return word_prob
    
    def get_glove_vectors(self, path):
        df = pd.read_csv(path, sep=' ', header=None, engine='c',
                         quoting=csv.QUOTE_ALL,error_bad_lines=False)
        token2id = {t:i  for i, t in enumerate(df.iloc[:, 0])}

        return df.iloc[:, 1:].values, token2id 

    def get_sentense_vectors(self, texts: List[List[str]]):
        li_sentense_vectors = []
        for text in texts:
            vectors = self._texts2vectors(text)
            tokens = [t for t in text if t in self.known_tokens]
            weight = [self.alpha/(self.word_prob[t] + self.alpha) for t in tokens]
            weight = np.diag(weight)
            vectors = np.mean(weight.dot(vectors),axis=0)
            vectors = vectors/np.linalg.norm(vectors)
            if np.isnan(vectors).any():
                continue
            li_sentense_vectors.append(vectors)

        sentense_vector_matrix = np.array(li_sentense_vectors).T
        u,_,_= svd(sentense_vector_matrix, )
        discourse = u[:, 0:1]
        proj_matrix = discourse.dot(discourse.T)
        sentense_vector_matrix = sentense_vector_matrix# - proj_matrix.dot(sentense_vector_matrix) 

        return sentense_vector_matrix


    def _texts2vectors(self, text: List[str]):
        tokens = [t for t in text if t in self.known_tokens]
        ids = [self.token2id.get(t) for t in tokens]
        vectors = self.vectors[ids]

        return vectors
        

class LSTMArcFace(nn.Module):
    def __init__(self, input_dim, latent_dim, label_dim):
        super(LSTMArcFace, self).__init__()
        print(label_dim)
        self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(latent_dim, label_dim)
        self.ArcFace = ArcMarginProduct(latent_dim, label_dim)
        self.loss = nn.CrossEntropyLoss()

    def _lstm_forward(self, input_sequence):
        lengths = [i.shape[0] for i in input_sequence]
        inputs = pad_sequence(input_sequence, batch_first=True)
        packed = pack_padded_sequence(inputs, lengths, batch_first=True)
        output, (h_t, c_t) = self.lstm(packed) 
        output, _ = pad_packed_sequence(output, batch_first=True)

        return h_t[0] 

    def forward(self, input_list, label):
        h = self._lstm_forward(input_list) 
        h = self.ArcFace(h, label)

        return h

    def get_loss(self, input_list, label):
        h = self.forward(input_list, label)
        loss = self.loss(h, label)

        return loss 

    