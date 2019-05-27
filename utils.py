import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import StratifiedKFold

# nltk.download('stopwords')
# nltk.download('punkt')
def load_data():
    df = pd.read_csv('./dataset.csv', engine='python')
    df['texts'] = df['news'].apply(lambda x: x.replace("'",'').replace(".",'\n').split('\n'))

    return df


def tokenize(df):
    df = df.copy()
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words += ["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s","``"]

    def _delete_stopwords(texts):
        li_tokens = []
        for text in texts:
            tokens = word_tokenize(text)
            li = [t for t in tokens if len(t) != 1]
            li = [t.lower() for t in li]
            li = [t for t in li if t not in stop_words]
            if len(li) != 0:
                li_tokens.append(li)

        return li_tokens

    df['tokenized'] = df['texts'].apply(_delete_stopwords)

    return df


def add_validation_label(df):
    df = df.copy()
    tags = df['type'].drop_duplicates()
    labels = {tag: i for i,tag in enumerate(tags)}
    df['label'] = df['type'].apply(lambda x: labels[x])
    skf = StratifiedKFold(n_splits=5)
    
    val_index = np.zeros(df.shape[0], dtype=int)
    for k, (_, val) in enumerate(skf.split(df['news'], df['label'])):
        val_index[val] =  k

    df['validation'] =  val_index

    return df