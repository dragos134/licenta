import re
import numpy
import gzip
import pickle
import time
import random
from nltk.corpus import wordnet
from sklearn import svm
from sklearn import naive_bayes


DATASET = './dataset/dataset.pkl.gz'
GRAM_FILE = '-gram.txt'
GRAM_PKL = '-gram.pkl.gz'
MAX_GRAM = 7

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
    
    rand_neigh = minority[list(sorted(distances, key=lambda x: x[0], reverse=True))[random.randint(1, k)][1]]

    dif_poly = poly_dif(rand_start, rand_neigh)

    max_dif = dif_poly[0]
    max_index = 0
    for i in range(n):
        if dif_poly[i] > max_dif:
            max_index = i
            max_dif = dif_poly[i]

    # print(f'punctul de start este {max_index}, cu diferenta {max_dif}')
    start_point = random.randint(int(min(rand_start[max_index], rand_neigh[max_index])), int(max(rand_start[max_index], rand_neigh[max_index])))

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
            ret_list[(curr_index + 1) % n] = random.randint(int(min(rand_start[max_index], rand_neigh[max_index])), int(max(rand_start[max_index], rand_neigh[max_index])))

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
    for i in range(len(true_set[0]), len(false_set[0])):
        count += 1
        start_time = time.time()
        true_set[0].append(smote_entry(true_set[0]))
        print('%.2f' % (time.time() - start_time))
        time_vec.append(time.time() - start_time)
        true_set[1].append(1)
    print(sum(time_vec) / count)
    
    new_train = [true_set[0] + false_set[0], true_set[1] + false_set[1]]

    new_train = list(zip(new_train[0], new_train[1]))
    random.shuffle(new_train)
    new_train = list(zip(*new_train))
    return new_train

def n_gram(n : int, train : list, test : list, firstx : int):
    '''
    Functie care scrie in fisiere n-gramele diferite
    '''

    n_gram_dict = dict()

    tweets = train[0]

    if n > MAX_GRAM:
        n = MAX_GRAM
    
    # scriem n-gramele in fisier
    for tweet in tweets:                                    # iteram prin fiecare tweet
        tweet = tweet.split(' ')                            # impartim tweetul in cuvinte
        for j in range(len(tweet) - n + 1):                 # iteram prin fiecare cuvant pana la lungimea tweetului - n
            curr_load = []
            for k in range(n):                              # iteram prin urmatoarele n cuvinte
                curr_load.append(tweet[j + k])              # adaugam in vectorul de n-grame
            curr_load = ' '.join(curr_load)                 # transformam vectorul inapoi in string pentru o mai usoara manevrare    
            if curr_load not in n_gram_dict:                    # verificam daca avem n-grama noua                   
                n_gram_dict[curr_load] = 1
            else:
                n_gram_dict[curr_load] += 1

    my_dict = [k for k, v in sorted(n_gram_dict.items(), reverse=True, key=lambda item: item[1])][:firstx]
    n_gram_file = open(f"./n_gram/{n}{GRAM_FILE}", 'w')
    n_gram_file.write('\n'.join(my_dict))
    n_gram_file.close()
    print(len(my_dict))

    # transformam datele in vectori de 0 si 1
    train_n_gram = []
    n_gram_size = len(my_dict)
    for tweet in tweets:
        tweet = tweet.split(' ')
        curr_load = [0] * n_gram_size
        for j in range(len(tweet) - n + 1):
            curr_n_word = []
            for k in range(n):
                curr_n_word.append(tweet[j + k])
            curr_n_word = ' '.join(curr_n_word)
            if curr_n_word in my_dict:
                curr_load[my_dict.index(curr_n_word)] += 1
        train_n_gram.append(curr_load)
    new_train = [train_n_gram, train[1]]

    print('incepe balansarea datelor')
    # new_train = smote_balance(new_train)

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
                curr_load[my_dict.index(curr_n_word)] += 1
        test_n_gram.append(curr_load)
    new_test = [test_n_gram, test[1]]

    n_gram_file = open(f"./n_gram/{n}{GRAM_PKL}", "wb")
    pickle.dump([new_train, new_test], n_gram_file)
    n_gram_file.close()

def SVM_train_n_gram(n : int):
    f = open(f"./n_gram/{n}{GRAM_PKL}", "rb")
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
    for i in range(MAX_GRAM):
        print(f'a inceput seria {i+1} de scris in fisier')
        n_gram(i + 1, train_set, test_set, 20000)
        print(f's-a terminate seria {i + 1}')

    # partea de antrenat cu SVM
    # for i in range(MAX_GRAM):
    #     SVM_train_n_gram(i + 1)
    #     print(f"s-a terminat seria {i + 1}")

    # regresie logistica
