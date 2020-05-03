import re
import numpy
import gzip
import pickle
import time
import random
from nltk.corpus import wordnet
from sklearn import svm
from sklearn import naive_bayes


DATASET = '../dataset.pkl.gz'
GRAM_FILE = '-gram.txt'
GRAM_PKL = '-gram.pkl.gz'
MAX_GRAM = 2

def n_gram(n : int, train : list, test : list):
    '''
    Functie care scrie in fisiere n-gramele diferite
    '''

    tweets = train[0]

    my_dict = []

    if n > MAX_GRAM:
        n = MAX_GRAM
    
    # scriem n-gramele in fisier
    n_gram_file = open(f"{n}{GRAM_FILE}", 'w')
    for i, tweet in enumerate(tweets):
        tweet = tweet.split(' ')
        for j in range(len(tweet) - n + 1):
            curr_load = []
            for k in range(n):
                curr_load.append(tweet[j + k])
            curr_load = ' '.join(curr_load)
            if curr_load not in my_dict:
                my_dict.append(curr_load)
                n_gram_file.write(curr_load + '\n')
    n_gram_file.close()

    # transformam datele in vectori de 0 si 1
    train_n_gram = []
    n_gram_size = len(my_dict)
    for i, tweet in enumerate(tweets):
        tweet = tweet.split(' ')
        curr_load = [0] * n_gram_size
        for j in range(len(tweet) - n + 1):
            curr_n_word = []
            for k in range(n):
                curr_n_word.append(tweet[j + k])
            curr_n_word = ' '.join(curr_n_word)
            curr_load[my_dict.index(curr_n_word)] = 1
        train_n_gram.append(curr_load)
    new_train = [train_n_gram, train[1]]

    test_n_gram = []
    tweets = test[0]
    for i, tweet in enumerate(tweets):
        tweet = tweet.split(' ')
        curr_load = [0] * n_gram_size
        for j in range(len(tweet) - n + 1):
            curr_n_word = []
            for k in range(n):
                curr_n_word.append(tweet[j + k])
            curr_n_word = ' '.join(curr_n_word)
            if  curr_n_word in my_dict:
                curr_load[my_dict.index(curr_n_word)] = 1
        test_n_gram.append(curr_load)
    new_test = [test_n_gram, test[1]]

    n_gram_file = open(f"{n}{GRAM_PKL}", "wb")
    pickle.dump([new_train, new_test], n_gram_file)
    n_gram_file.close()

def SVM_train_n_gram(n : int):
    f = open(f"{n}{GRAM_PKL}", "rb")
    train, test = pickle.load(f)
    f.close()
    clf = svm.SVC()
    
    print('se incepe antrenamentul...')
    clf.fit(train[0], train[1])

    print('se face prezicerea...')
    prediction = clf.predict(test[0])

    print('se verifica procentajul...')
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == 1:
            print(test[1][i])
        if prediction[i] == test[1][i]:
            count += 1
    print(count * 100 / len(test[1]))

if __name__ == "__main__":
    f = gzip.open(DATASET, 'rb')
    train_set, test_set = pickle.load(f)
    f.close()
    # partea de scris in fisiere
    # for i in range(MAX_GRAM):
    #     n_gram(i + 1, train_set, test_set)
    #     print(f's-a terminate seria {i + 1}')

    # partea de antrenat cu SVM
    for i in range(MAX_GRAM):
        SVM_train_n_gram(i + 1)
        print(f"s-a terminat seria {i + 1}")
