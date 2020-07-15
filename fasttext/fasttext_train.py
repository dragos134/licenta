import fasttext as ft
import gzip
import pickle
import time
import copy
import random
from sklearn import svm, preprocessing
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
import wandb
from wandb.keras import WandbCallback


BLNC = 70



def calculate_distance(a : list, b : list):
    my_sum = 0
    for i in range(len(a)):                                             # calculam distanta
        my_sum = my_sum + (a[i] - b[i]) ** 2                            

    return my_sum ** (1 / 2)

def poly_dif(a : list, b : list) -> list:
    ret_dif = []
    for i in range(len(a)):
        ret_dif.append(a[i] - b[i])
    return ret_dif

def smote_entry(minority : list, k=5) -> list:
    n = len(minority[0])
    rand_start = minority[random.randint(0, len(minority) - 1)]
    distances = []
    for i, entry in enumerate(minority):
        distances.append((calculate_distance(entry, rand_start), i))
    
    rand_neigh = minority[list(sorted(distances, key=lambda x: x[0]))[random.randint(1, k)][1]]

    dif_poly = poly_dif(rand_start, rand_neigh)

    max_dif = dif_poly[0]
    max_index = 0
    for i in range(n):
        if dif_poly[i] > max_dif:
            max_index = i
            max_dif = dif_poly[i]

    # print(f'punctul de start este {max_index}, cu diferenta {max_dif}')
    my_min = min(rand_start[max_index], rand_neigh[max_index])
    my_max = max(rand_start[max_index], rand_neigh[max_index])
    start_point = random.random() * (my_max - my_min) + my_min
    ret_list = [0] * n

    ret_list[max_index] = start_point

    for i in range(n - 1):
        curr_index = (max_index + i) % n
        if dif_poly[curr_index] * dif_poly[(curr_index + 1) % n] > 0:
            a = rand_neigh[curr_index] * dif_poly[(curr_index + 1) % n]
            c = -rand_neigh[(curr_index + 1) % n] * dif_poly[curr_index]
            aux = [dif_poly[(curr_index + 1) % n], -dif_poly[curr_index]]
        else:
            a = rand_neigh[curr_index] * dif_poly[(curr_index + 1) % n]
            c = rand_neigh[(curr_index + 1) % n] * dif_poly[curr_index]
            aux = [dif_poly[(curr_index + 1) % n], dif_poly[curr_index]]
        
        if aux[1] != 0:
            ret_list[(curr_index + 1) % n] = (a + c - ret_list[curr_index] * aux[0]) / aux[1]
        else:
            my_min = min(rand_start[max_index], rand_neigh[max_index])
            my_max = max(rand_start[max_index], rand_neigh[max_index])
            ret_list[(curr_index + 1) % n] = random.random() * (my_max - my_min) + my_min

    return ret_list

        

    





    

        


def smote_balance(train : list) -> list:
    true_set = [[], []]
    false_set = [[], []]

    for i in range(len(train[0])):                                      #separam datele true de cele false
        if train[1][i] == 1:
            true_set[0].append(train[0][i])
            true_set[1].append(train[1][i])
        else:
            false_set[0].append(train[0][i])
            false_set[1].append(train[1][i])

    count = 0
    time_vec = []
    print(len(true_set[0]), len(false_set[0]))
    lt = len(true_set[0])
    lf = len(false_set[0])
    for i in range(len(true_set[0]), len(false_set[0])):
        count += 1
        start_time = time.time()
        true_set[0].append(smote_entry(true_set[0]))
        # print('%.2f' % (time.time() - start_time))
        time_vec.append(time.time() - start_time)
        true_set[1].append(1)
        print('%.2f la suta completed, finishing in %sm' % (100 * (i - lt) / (lf - lt), int((lf - i) * (sum(time_vec) / count) / 60)), end='\r')

    print(sum(time_vec) / count)
    
    new_train = [true_set[0] + false_set[0], true_set[1] + false_set[1]]

    new_train = list(zip(new_train[0], new_train[1]))
    random.shuffle(new_train)
    new_train = list(zip(*new_train))
    return new_train

def fasttext_raw_prep():
    f = open('./dataset/optimized_tweets.txt', 'r')
    file_content = f.read().split('\x01')
    f.close()

    f = open('./fasttext/raw_data.txt', 'w')
    for tweet in file_content:
        f.write(tweet.split('\x02')[0] + '\n')
    f.close()

def fasttext_preprocessed():
    f = gzip.open(f'./dataset/dataset.pkl.gz', 'rb')
    train_set, test_set = pickle.load(f)
    f.close()
    f = open('./fasttext/processed_data.txt', 'w')
    for tweet in train_set[0]:
        f.write(tweet + '\n')
    f.close()

def append_train_to_corpora():
    f = open('./fasttext/processed_data.txt', 'r')
    corpora_f = open('./twitter_corpora/corpora.txt', 'a')
    corpora_f.write(f.read())
    corpora_f.close()
    f.close()

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

def vec_sum(v : list) -> list:
    word_len = len(v[0])
    ret_list = [0] * word_len
    for i in v:
        for j in range(word_len):
            ret_list[j] += i[j]
    return ret_list

def test(test_output : list, prediction : list) -> float:
    count = 0
    for i in range(len(prediction)):
        if test_output[i] == prediction[i]:
            count += 1
    return count * 100 / len(prediction)

def f_score(test_output : list, prediction : list) -> float:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    print(len(prediction), len(test_output))
    total_true = 0
    total_false = 0
    for i in test_output:
        if i == 0:
            total_false += 1
        else:
            total_true += 1
    for i in range(len(prediction)):
        if test_output[i] == 1:
            if prediction[i] == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if prediction[i] == 1:
                false_positives += 1
            else:
                true_negatives += 1

    print(f'true_positives: {true_positives}\ttrue_negatives: {true_negatives}\tfalse_positives: {false_positives}\tfalse_negatives: {false_negatives}')

    p = true_positives / (true_positives + false_positives)
    r = true_positives / (true_positives + false_negatives)

    return (2 * p * r) / (p + r)


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

    for i in train_set[0]:
        if len(i) != len(train_set[0][0]):
            print(i)
            print('a')

    for i in train_set[1]:
        if type(i) != int:
            print('b')

    clf.fit(train_set[0], train_set[1])

    best_clf = clf

    print("Se face prezicerea pe setul de test")
    prediction = clf.predict(test_set[0])

    proc1 = test(test_set[1], prediction)
    print(proc1)

    best_fscore = f_score(test_set[1], prediction)
    print('s-a obtinut fscorul %.3f' % best_fscore)


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
    while i < 10 and proc < 99:
        print(f"Epoca {i+1}: ", end='')
        clf.fit(train[0], train[1])
        prediction = clf.predict(valid[0])
        proc = test(valid[1], prediction)
        fscore = f_score(valid[1], prediction)
        print("%.3f" % proc)
        print("f-score-ul pentru clasa 1 este %.3f" % (fscore))
        prediction = clf.predict(test_set[0])
        print("f-score-ul pentru setul de test este %.3f" % (f_score(test_set[1], prediction)))
        if fscore > best_fscore:
            best_fscore = fscore
            best_clf = clf

        i += 1

        print("Se amesteca datele")
        print(len(train[0]), len(valid[0]))
        train, valid = cross_validation(train, valid)

    prediction = best_clf.predict(test_set[0])
    proc2 = test(test_set[1], prediction)
    print("%.2f" % proc2)
    print("f-score-ul pentru clasa 1 este %.2f" % (f_score(test_set[1], prediction)))

    return max(proc1, proc2)

def neural_net(train, test):
    wandb.init(project="bachelor")
    keras_model = Sequential()
    keras_model.add(Dense(len(train[0][0]), activation="relu"))
    keras_model.add(Dense(50, activation="relu"))
    keras_model.add(Dense(1, activation="sigmoid"))

    keras_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()])
    keras_model.fit(train[0], train[1], batch_size=3, epochs=20, callbacks=[WandbCallback()])
    loss, acc, tp, fp, tn, fn = keras_model.evaluate(test[0], test[1], callbacks=[WandbCallback()])

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    print(f'f-score is: {(2 * p * r) / (p + r)}')

def train_fasttext_model():
    model = ft.train_unsupervised('./twitter_corpora/corpora.txt', model='skipgram', dim=45)
    model.save_model('./fasttext/sk_fasttext.bin')

    model = ft.train_unsupervised('./twitter_corpora/corpora.txt', model='cbow', dim=45)
    model.save_model('./fasttext/cbow_fasttext.bin')

def fasttexting(filename):
    model = ft.load_model('./fasttext/ft_model.bin')

    f = gzip.open(f'./dataset/{filename}.pkl.gz')
    train_set, test_set = pickle.load(f)
    train_set[0] = list(train_set[0])
    train_set[1] = list(train_set[1])
    test_set[0] = list(test_set[0])
    test_set[1] = list(test_set[1])
    f.close()

    for i, tweet in enumerate(train_set[0]):
        avg = []
        tweet = tweet.split(' ')
        for word in tweet:
            avg.append(model[word])
        train_set[0][i] = vec_sum(avg)

    for i, tweet in enumerate(test_set[0]):
        avg = []
        tweet = tweet.split(' ')
        for word in tweet:
            avg.append(model[word])
        test_set[0][i] = vec_sum(avg)

    f = gzip.open(f'./fasttext/{filename}.pkl.gz', 'wb')
    pickle.dump([train_set, test_set], f)
    f.close()

if __name__ == "__main__":

    # fasttexting('12dataset')
    f = gzip.open(f'./fasttext/12dataset.pkl.gz', 'rb')
    train_set, test_set = pickle.load(f)
    f.close()

    neural_net(train_set, test_set)

    clf = svm.SVC(C=2, kernel='rbf')
    print(train(train_set, test_set, clf))

    clf = GaussianNB()
    print(train(train_set, test_set, clf))


    clf = LogisticRegression(warm_start=True, max_iter=200)
    print(train(train_set, test_set, clf))