from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import jieba
import re


def get_title_context():
    """
    将文档进行分词，返回结果，用空格分隔；
    :return:
    """
    with open('SogouCS.WWW08.txt') as f:
        file = f.read()
        file_string = ""
        for i in file:
            file_string += i
        document = re.compile('<doc>' + '(.*?)' + '</doc>', re.S)
        title = re.compile('<contenttitle>' + '(.*?)' + '</contenttitle>', re.S)
        context = re.compile('<content>' + '(.*?)' + '</content>', re.S)
        documents = document.findall(file_string)
        titles, contexts = [], []
        for lines in documents:
            titles_tmp = title.findall(lines)
            contexts_tmp = context.findall(lines)
            contexts_tmp = str(contexts_tmp).replace('\\n', '')
            titles.append(titles_tmp)
            contexts.append(contexts_tmp)
        c = []
        for item in contexts:
            tmp = jieba.cut(item,cut_all=False)
            c.append(' '.join(tmp))
    return titles, c, documents


def transform(dataset, n_features):
    """
    :param n_features: 从文档中提取出tfidf值排名前n_features的单词
                       将每篇文档都表示成n_features的向量，如不含该词则对应的值为0；
    """
    vectorizer = TfidfVectorizer(max_df=100, max_features=n_features, min_df=10, use_idf=True,
                                 dtype=float)
    X = vectorizer.fit_transform(dataset)
    # print(X.shape)
    # vectorizer.get_feature_names()返回满足结果的前n_features个词;
    # print(vectorizer.get_feature_names())

    return X, vectorizer


def train(X, vectorizer, true_k=10, minibatch=False, showLable=False):
    # 使用采样数据还是原始数据训练k-means，
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                verbose=False)
    km.fit(X)
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(km.predict(X))
    # print(result)
    print('Cluster distribution:')
    print(dict([(i, result.count(i)) for i in result]))
    return -km.score(X)


def test():
    '''测试选择最优参数'''
    title, dataset, document = get_title_context()
    # dataset = loadDataset()
    print("%d documents" % len(dataset))
    X, vectorizer = transform(dataset, n_features=500)
    true_ks = []
    scores = []
    # for i in range(3, 80, 1):
    #     score = train(X, vectorizer, true_k=i) / len(dataset)
    #     # print(i, score)
    #     true_ks.append(i)
    #     scores.append(score)
    # plt.figure(figsize=(8, 4))
    # plt.plot(true_ks, scores, label="error", color="red", linewidth=1)
    # plt.xlabel("n_features")
    # plt.ylabel("error")
    # plt.legend()
    # plt.show()


def out():
    '''在最优参数下输出聚类结果'''
    title, dataset, document = get_title_context()
    X,vectorizer = transform(dataset,n_features=500)
    score = train(X,vectorizer,true_k=12,showLable=True)/len(dataset)
    print (score)

# test()
out()
# title, context, document = get_title_context()
# X, vectorizer = transform(context)
