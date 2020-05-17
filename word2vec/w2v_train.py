import gensim.downloader as api
import gensim.models
from gensim import utils
from gensim.test.utils import datapath
import pickle
import gzip
import time
import copy
import random
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

BLNC = 70

def test(test_output : list, prediction : list) -> float:
    count = 0
    proc = 0
    for i in range(len(prediction)):
        if test_output[i] == 1:
            count += 1
            if test_output[i] == prediction[i]:
                proc += 1
    return proc * 100 / count


def cross_validation(train_set : list, valid_set : list) -> tuple:

    cpy = list(zip(train_set[0] + valid_set[0], train_set[1] + valid_set[1]))
    random.shuffle(cpy)

    train = cpy[:int(BLNC * len(cpy) / 100)]
    random.shuffle(train)
    train = list(zip(*train))

    valid = cpy[int(BLNC * len(cpy) / 100):]
    random.shuffle(valid)
    valid = list(zip(*valid))

    return (train, valid)

def train(train_set : list, test_set : list, my_clf) -> float:
    #regular train
    # clf = svm.SVC(max_iter=10)
    print("Se incepe antrenamentul normal")
    clf = copy.deepcopy(my_clf)
    clf.fit(train_set[0], train_set[1])

    print("Se face prezicerea pe setul de test")
    prediction = clf.predict(test_set[0])

    proc1 = test(test_set[1], prediction)
    print(proc1)


    #cross validation fit
    print("Incepe antrenanemtul cross-validation")
    train = list(zip(train_set[0], train_set[1]))
    valid = train[int(BLNC * len(train) / 100):]
    valid = list(zip(*valid))
    train = train[:int(BLNC * len(train) / 100)]
    train = list(zip(*train))
    clf = copy.deepcopy(my_clf)
    i = 0
    proc = 0
    while i < 10 and proc < 90:
        print(f"Epoca {i+1}: ", end='')
        clf.fit(train[0], train[1])
        prediction = clf.predict(valid[0])
        proc = test(valid[1], prediction)
        print(proc)

        i += 1

        print("Se amesteca datele")
        print(len(train[0]), len(valid[0]))
        train, valid = (train, valid)

    prediction = clf.predict(test_set[0])
    proc2 = test(test_set[1], prediction)
    print(proc2)

    return max(proc1, proc2)
    

def vec_avg(v : list) -> list:
    vec_nr = len(v)
    # print(vec_nr)
    ret_vec = v[0].copy()
    for i in range(1, vec_nr):
        for j in range(len(v[i])):
            ret_vec[j] += v[i][j]
    for i in range(len(ret_vec)):
        ret_vec[i] = ret_vec[i] / vec_nr
    return ret_vec

if __name__ == "__main__":
    f = gzip.open("dataset.pkl.gz", "rb")
    train_set, test_set = pickle.load(f)
    f.close()
    train_set[0] = list(train_set[0])
    test_set[0] = list(test_set[0])
    for i, tweet in enumerate(train_set[0]):
        train_set[0][i] = gensim.utils.simple_preprocess(tweet)
    for i, tweet in enumerate(test_set[0]):
        test_set[0][i] = gensim.utils.simple_preprocess(tweet)
    model = gensim.models.Word2Vec(sentences=(train_set[0] + gensim.test.utils.common_texts), min_count=2, size=40, iter=10)
    model.train(sentences=train_set[0], total_examples=model.corpus_count, epochs=30)

    for i, tweet in enumerate(train_set[0]):
        avg = []
        for word in tweet:
            if word in model.wv:
                avg.append(model.wv[word])
        train_set[0][i] = vec_avg(avg)

    for i, tweet in enumerate(test_set[0]):
        avg = []
        for word in tweet:
            if word in model.wv:
                avg.append(model.wv[word])
        test_set[0][i] = vec_avg(avg)

    clf = svm.SVC(max_iter=10)
    print(train(train_set, test_set, clf))

    clf = RandomForestClassifier(n_estimators=20)
    print(train(train_set, test_set, clf))

    clf = AdaBoostClassifier(n_estimators=100)
    print(train(train_set, test_set, clf))

    clf = GradientBoostingClassifier(n_estimators=100)
    print(train(train_set, test_set, clf))


    
