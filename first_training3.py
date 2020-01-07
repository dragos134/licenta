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

f = gzip.open('new_data_set.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()



clf = svm.SVC()

clf.fit(train_set[0][:10], train_set[1][:10])
print('s-a terminat de antrenat')

predic = clf.predict(test_set[0])
print('s-a terminat de prezis')

count = 0
for i in range(len(predic)):
    if predic[i] == 1:
        print(count)
    if predic[i] == test_set[1][i]:
        count += 1

print(count * 100 / len(test_set[1]))


