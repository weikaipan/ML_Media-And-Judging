"""utilities."""

import re
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

import time
import math
import unicodedata
import json
import datetime
import pickle


def formatDate(d, f):
    return datetime.datetime.strptime(d, f).strftime("%Y%m%d")


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def remove_non_ascii(text):
    """
    Note:
     ord('a') returns the integer 97 , ord(u'\u2020') returns 8224 .
    """
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = remove_non_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?']+", r" ", s)
    return str(s.encode('utf-8').decode('ascii', 'ignore'))


def clean_text(text):
    text.replace("LEAD:", "")
    return text.replace("lead:", "")


def text_dump(obj, file_path):
    with open(file_path, 'w') as fout:
        fout.write(obj)


def json_load(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def json_dump(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def gettime(start):
    now = time.time()
    return asMinutes(now - start)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def pickle_dump(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def to_one_hot(df):
    onehot = df.select_dtypes(include=[object])
    X = pd.get_dummies(onehot)
    df = df.drop(onehot.columns.values, axis=1)
    enc = OneHotEncoder(sparse=False)
    uniq_vals = X.apply(lambda x: x.value_counts()).unstack()
    uniq_vals = uniq_vals[~uniq_vals.isnull()]
    enc_cols = list(uniq_vals.index.map('{0[0]}_{0[1]}'.format))
    X_transform = enc.fit_transform(X.as_matrix())
    enc_df = pd.DataFrame(X_transform, columns=enc_cols, index=X.index)
    df = pd.concat([df, enc_df], axis=1)
    return df


def concat_dataframe(df1, df2):
    """This function horizontally concates two dataframe"""
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    return pd.concat([df1, df2], axis=1)


def keep_noun_verb_adjs(text):
    tokens = word_tokenize(text.lower())
    tags = pos_tag(tokens)
    text_list = [word for word, pos in tags if (pos == 'NN' or
                                                pos == 'NNP' or pos == 'NNS' or
                                                pos == 'NNPS' or pos == 'VB' or
                                                pos == 'VBD' or pos == 'VBG' or
                                                pos == 'VBN' or pos == 'VBP' or
                                                pos == 'VBZ' or pos == 'JJ' or
                                                pos == 'JJR' or pos == 'JJS')]
    return " ".join(text_list)
