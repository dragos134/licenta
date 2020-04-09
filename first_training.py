
import sys
import copy
import time
import re
import numpy
import gzip
import pickle
from sklearn import svm
from nltk.corpus import wordnet


resource_file = open('tweets_text.txt')

text = resource_file.read()

new_text = re.findall('[0-9]+\t[^\t]+\t[01]', text)

all_set = []

train_set = []
valid_set = []
train_set = []

# for line in resource_file:
#     all_set.append(line.split('\t'))

dictionary = []

for i in new_text:
    all_set.append(i.split('\t'))

# dict_file = open('dict.txt', 'w')

for i, tweet in enumerate(all_set):
    all_set[i][1] = tweet[1].split(' ')
    for j, word in enumerate(all_set[i][1]):
        all_set[i][1][j] = word.replace(',', '').replace('.', '').replace('?', '').replace('!', '').replace('\n', '')
        if all_set[i][1][j] not in dictionary:
            dictionary.append(all_set[i][1][j])
            # dict_file.write(all_set[i][1][j] + '\n')

# dict_file.close()


data_set = []
outputs = []

for i, tweet in enumerate(all_set):
    data_set.append([])
    outputs.append(tweet[2])
    for j, word in enumerate(dictionary):
        if word in tweet[1]:
            data_set[i].append(1)
        else:
            data_set[i].append(0)




# f = gzip.open('data_set.pkl.gz', 'wb')

# pickle.dump([data_set, outputs], f)

# f.close()




true_set = []
false_set = []

for i in range(len(data_set)):
    if outputs[i] is '1':
        true_set.append([data_set[i], 1])
    else:
        false_set.append([data_set[i], 0])
# true_set = numpy.array(true_set)
# false_set = numpy.array(false_set)

# new_file = open('optimized_tweets.txt', 'w')

# for i in all_set:
#     new_file.write(i[1] + '##' + i[2] + '$#')

# new_file.close()

a = []
b = []
c = []
d = []
e = []
f = []

train_set = true_set[:((70 * len(true_set)) // 100)] + false_set[:((70 * len(false_set)) // 100)]
numpy.random.shuffle(train_set)
print('s-a construit train_set')

for tweet in train_set:
    a.append(tweet[0])
    b.append(tweet[1])

print('s-a pregatit train_set pentru pickle')

valid_set = true_set[((70 * len(true_set)) // 100):((85 * len(true_set)) // 100)] + false_set[((70 * len(false_set)) // 100):((85 * len(false_set)) // 100)]
numpy.random.shuffle(valid_set)

print('s-a construit valid_set')

for tweet in valid_set:
    c.append(tweet[0])
    d.append(tweet[1])

print('s-a pregatit valid_set pentru pickle')

test_set = true_set[((85 * len(true_set)) // 100):] + false_set[((85 * len(false_set)) // 100):]
numpy.random.shuffle(test_set)
print('s-a construit test_set')

for tweet in test_set:
    e.append(tweet[0])
    f.append(tweet[1])
print('s-a pregatit test_set pentru pickle')

file = gzip.open('new_data_set.pkl.gz', 'wb')
pickle.dump(([a, b], [c, d], [e, f]), file)
print('s-au introdus datele in fisier')
file.close()



#distanta cosin (cosin similarity)
#word2vec