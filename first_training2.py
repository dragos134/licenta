import sys
import copy
import time
import re
import numpy
import gzip
import pickle
from nltk.corpus import wordnet
from sklearn import svm
from sklearn import naive_bayes

f = gzip.open('data_set.pkl.gz')
data_set = pickle.load(f)
f.close()

clf = svm.SVC()


for k in range(9):
    clf.fit(data_set[0][1000 * k : 1000 * (k + 1)], data_set[1][1000 * k : 1000 * (k + 1)])
    print(k)

print('s-a terminat de antrenat')

# clf = naive_bayes.BernoulliNB()
# clf.fit(data_set[0], data_set[1])

count = 0



for i in range(len(data_set[0])):
    if clf.predict([data_set[0][i]]) == data_set[1][i]:
        count += 1

print(count * 100 / len(data_set[0]))
