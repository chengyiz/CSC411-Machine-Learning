# -*- decoding: utf-8 -*-
'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from tabulate import tabulate

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train loss = {}'.format(1-(train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test loss = {}'.format(1-(test_pred == test_labels).mean()))
    return model

def bnb_perceptron(bow_train, train_labels, bow_test, test_labels):
    avg_accus =[]
    kf = KFold(n_splits=10)
    print('='*80)
    print('Perceptron')
    rang = [5,10,20,30,40,50,60,70,80,90,100]
    for i in rang:
        print('Doing: ' + str(i))
        accus=[]
        for train_index, test_index in kf.split(bow_train):
            trains, tests = bow_train[train_index], bow_train[test_index]
            label_train, label_test = train_labels[train_index], train_labels[test_index]
            
            model = Perceptron(max_iter=i)
            model.fit(trains, label_train)
            
            pred = model.predict(tests)
            accus.append(metrics.accuracy_score(pred, label_test))
        avg_accus.append(sum(accus)/ float(len(accus)))
    print('10-Fold accuracies:')
    print(avg_accus)
    max_i = rang[avg_accus.index(max(avg_accus))]
    print('Best max_iter number for Perceptron is: ' + str(max_i))
    model = Perceptron(max_iter = max_i)
    model.fit(bow_train, train_labels)
    train_pred = model.predict(bow_train)
    print('Perceptron train loss = {}'.format(1-metrics.accuracy_score(train_pred, train_labels)))
    test_pred = model.predict(bow_test)
    print('Perceptron test loss = {}'.format(1-metrics.accuracy_score(test_pred, test_labels)))
    return model
    
def bnb_multinomial(bow_train, train_labels, bow_test, test_labels):
    print('='*80)
    print('MultinomialNB')
    avg_accus=[]
    kf = KFold(n_splits=10)
    rang = [i * 0.1 for i in range(11)]
    for i in rang:
        print('Doing alpha=',str(i))
        accus=[]
        for train_index, test_index in kf.split(bow_train):
            trains, tests = bow_train[train_index], bow_train[test_index]
            label_train, label_test = train_labels[train_index], train_labels[test_index]
            
            model = MultinomialNB(alpha=i)
            model.fit(trains, label_train)
            
            pred = model.predict(tests)
            accus.append(metrics.accuracy_score(pred, label_test))
        avg_accus.append(sum(accus)/float(len(accus)))
    print('10-Fold accuracies:')
    print(avg_accus)
    max_i = rang[avg_accus.index(max(avg_accus))]
    print('Best alpha value is: ', str(max_i))
    model = MultinomialNB(alpha=max_i)
    model.fit(bow_train, train_labels)
    train_pred = model.predict(bow_train)
    print('MultinomialNB train loss = {}'.format(1-metrics.accuracy_score(train_pred, train_labels)))
    test_pred = model.predict(bow_test)
    print('MultinomialNB test loss = {}'.format(1-metrics.accuracy_score(test_pred, test_labels)))
    return model

def bnb_SGD(bow_train, train_labels, bow_test, test_labels):
    avg_accus=[]
    kf = KFold(n_splits=10)
    print('=' * 80)
    print('SGDClassifier')
    for penalty in ["l1", "l2", "elasticnet"]:
        print('Doing with penalty: ', penalty)
        accus=[]
        for train_index, test_index in kf.split(bow_train):
            trains, tests = bow_train[train_index], bow_train[test_index]
            label_train, label_test = train_labels[train_index], train_labels[test_index]

            model = SGDClassifier(alpha=0.0001, max_iter=50, penalty=penalty)
            model.fit(trains, label_train)
            
            pred = model.predict(tests)
            accus.append(metrics.accuracy_score(pred, label_test))
        avg_accus.append(sum(accus)/float(len(accus)))
    print('10-Fold accuracies:')
    print(avg_accus)
    opt_penalty = ["l1", "l2", "elasticnet"][avg_accus.index(max(avg_accus))]
    print('optimal penalty function is ', opt_penalty)
    model = SGDClassifier(alpha=0.0001, max_iter=50, penalty=penalty)
    model.fit(bow_train, train_labels)
    train_pred = model.predict(bow_train)
    test_pred = model.predict(bow_test)
    print('SGDClassifier train loss = {0}\nSGDClassifier test loss = {1}'.format(\
        1-metrics.accuracy_score(train_pred, train_labels), \
        1-metrics.accuracy_score(test_pred, test_labels))) 
    return model

if __name__ == '__main__':
    train_data, test_data = load_data()
    
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    print(test_data.target_names)

    bnb_model1 = bnb_perceptron(train_bow, train_data.target, test_bow, test_data.target)
    bnb_model2 = bnb_multinomial(train_bow, train_data.target, test_bow, test_data.target)
    bnb_model3 = bnb_SGD(train_bow, train_data.target, test_bow, test_data.target)
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    
    pred = bnb_model2.predict(test_bow)
    max_side, i = max(max(pred), max(test_data.target)) + 1, 0
    table = np.zeros((max_side, max_side))
    while i < pred.shape[0]:
        table[pred[i], test_data.target[i]] += 1
        i += 1
    temp = np.zeros((max_side, max_side))
    for i in range(max_side):
        for j in range(max_side):
            temp[i,j] = float(table[i,j]) / sum(table[i,:])
    print(tabulate(temp))
            
    f = open('table.txt', 'w')
    f.write(tabulate(table, headers=test_data.target_names))
    f.close()