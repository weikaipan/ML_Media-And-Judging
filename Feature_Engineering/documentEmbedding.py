"""This file embeds news articles using fasttext."""
import numpy as np
import pandas as pd
import time

from nltk.tokenize import word_tokenize
from pickleFileIO import pickle_dump, pickle_load
from preparenews import Doc
from utility import concat_dataframe, gettime, asMinutes
from SETTINGS import GROUPING
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


def article_embedding(document, word2vec):
    """Concat the max and min vector"""
    dim = 100
    tokens = word_tokenize(document.lower())
    embedding_max = np.array([-999.0] * dim)
    embedding_min = np.array([999.0] * dim)
    for word in tokens:
        try:
            word_vector = np.array([float(i) for i in word2vec[word]])
            embedding_max = np.maximum(embedding_max, word_vector)
            embedding_min = np.minimum(embedding_min, word_vector)
        except KeyError:
            pass
    embedding = np.append(embedding_max, embedding_min)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def readWord2Vec(path):
    def parse_word2vec(word2vec):
        w2v = {}
        for wordvec in word2vec:
            w2v[wordvec[0]] = wordvec[1:-1]
        return w2v

    print("Writing word2vec pickle...")
    word2vec = []
    with open(path) as f:
        next(f)
        for line in f:
            word2vec.append(line.split(" "))
    word2vec = parse_word2vec(word2vec)
    return word2vec


def embedding(title2doc, word2vec):
    title2embedding = {}
    i = 0
    print("Total News = {}".format(len(title2doc)))
    print("Begin vectorize articles")
    for title, doc in title2doc.items():
        i = i + 1
        if i % 1000 == 0:
            print("Processing on article: ", i)
            print("Remaining {}".format(len(title2doc) - i))
        title2embedding[title] = article_embedding(doc['content'], word2vec)
    return title2embedding


def elbowmethod(embeds):
    import matplotlib.pyplot as plt
    k_selections = [k for k in range(100, 300, 20)]
    models = [KMeans(n_clusters=k, random_state=0).fit(embeds)
              for k in k_selections]
    scores = [m.score(embeds) for m in models]
    plt.plot(k_selections, scores)
    plt.xlabel('clusters')
    plt.ylabel('K means scores')
    plt.savefig('./elbowmethod.jpg')
    plt.show()
    quit()


def clustering(title2embeds, K=300, elbow=False):
    """
    This function generates a dataframe: [title, group], for each news maps to
    its group
    """
    print("Clustering..")
    embeds = []
    title_list = []
    for title in title2embeds:
        title_list.append(title)
        embeds.append(title2embeds[title].tolist())
    if elbow:
        elbowmethod(embeds)
    else:
        kmeans = KMeans(n_clusters=K, random_state=0).fit(embeds)
    grouped_news = pd.DataFrame(embeds)
    grouped_news['group'] = kmeans.labels_
    grouped_news['title'] = title_list
    return grouped_news


def doctovec(doc):
    print("Read word vectors")
    word2vec = readWord2Vec('/data/WorkData/media_and_judging/data/prepared/news8710.vec')
    title2embedding = embedding(doc.title2content, word2vec)
    print("Saving: title2embedding.p")
    pickle_dump(title2embedding, '/data/WorkData/media_and_judging/data/prepared/news_title2embedding8710.p')
    return title2embedding


def mapGroupTitle(court_news_list, NewsLang, grouped_news, K=300):
    title2group = {}
    group2titles = {}

    grouped_news = grouped_news[['group', 'title']]
    group = grouped_news['group'].tolist()
    title = grouped_news['title'].tolist()

    group_count = np.zeros((court_news_list.shape[0], K))
    for g, t in zip(group, title):
        title2group[t] = g
        if g in group2titles:
            group2titles[g].append(t)
        else:
            group2titles[g] = []

    newsindexstring = court_news_list['nytimes_90day_index'].values.tolist()
    # transform news title into the group it belongs to
    c = 0
    for court_news_index in newsindexstring:
        # get news index list for each court case (happened during 30 days)
        newsindexlist = court_news_index.split(':')[1:-1]
        for i in range(len(newsindexlist)):
            title = NewsLang.index2title[int(newsindexlist[i])]
            try:
                group = title2group[title]
                group_count[c][group] += 1
            except KeyError:
                print("At row {}, idx {}, key error".format(c, newsindexlist[i]))
        c += 1

    result = pd.DataFrame(group_count, columns=['newsgroup' + str(i) for i in range(1, K + 1)])
    result = concat_dataframe(result, court_news_list)
    result.to_csv('/data/WorkData/media_and_judging/data/train/final_grouped500_news_court_trend_0110_test.csv')
    return


def title2bigram(doc, news_list, max_features=None):
    """Create a bigram count (for news title) array."""
    # Preload all title.
    print("Loading all title")
    title_text = []
    for title in doc.title2index:
        title_text.append(title)
    print("Load total: {}".format(len(title_text)))

    # Bigrams.
    # For
    vec = CountVectorizer(lowercase=True, stop_words='english',
                          max_features=max_features)
    vec.fit_transform(title_text)
    ngram_columns = vec.get_feature_names()
    ngram2idx = {ngram_columns[i]: i for i in range(len(ngram_columns))}

    # print(len(ngram_columns))
    # print(bigram2idx)
    # For each news title of court case, find if bigrams existing
    start = time.time()
    ngram_array = np.zeros((news_list.shape[0], len(ngram_columns)))
    case_number = 0
    for titlelist in news_list['nytimes_90day_index']:
        court_vec = CountVectorizer(lowercase=True, stop_words='english')
        idxs = titlelist.split(':')
        titles_percase = []
        # Aggregate all titles for this court case
        for idx in idxs:
            try:
                i = int(idx)
                titles_percase.append(doc.index2title[i])
            except:
                pass
        ngram_percase = court_vec.fit_transform(titles_percase)
        ngram_percase_column = court_vec.get_feature_names()
        ngram2idx_percase = {ngram_percase_column[i]: i
                             for i in range(len(ngram_percase_column))}
        # Add
        for ngrams in ngram_percase_column:
            try:
                ngram_array[case_number, ngram2idx[ngrams]] += ngram_percase[ngram2idx_percase[ngram_percase]]
            except:
                pass
        # print(bigram_columns)
        # print(bigram_array[case_number])
        # quit()
        case_number += 1
        if case_number % 2000 == 0:
            print(case_number)
            print(gettime(start))
    df = pd.DataFrame(ngram_array, columns=[ngram_array])
    df.to_csv('/data/WorkData/media_and_judging/data/featurized/court_news_bigram_list_90days.csv')
    return


def main():
    """Main Driver."""
    doc = Doc()
    doc = pickle_load('/data/WorkData/media_and_judging/data/prepared/newsobject8710.p')
    court_news_list = pd.read_csv('/data/WorkData/media_and_judging/data/prepared/court_news_list_90days_0110.csv')
    # title2embedding = doctovec(doc)
    title2embedding = pickle_load('/data/WorkData/media_and_judging/data/prepared/news_title2embedding8710.p')
    grouped_news = clustering(title2embedding, K=1, elbow=False)
    # grouped_news = pd.read_csv('../../DocumentEmbedding/news8710/grouped_news_df_8710.csv')
    if GROUPING:
        mapGroupTitle(court_news_list, doc, grouped_news, K=500)
    else:
        court_news_list = court_news_list[['nytimes_90day_index']]
        title2bigram(doc, court_news_list, max_features=200)


if __name__ == '__main__':
    main()
