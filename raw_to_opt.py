import re
import sys
import gzip
import pickle
import re
import time
import copy
import random

TXT_SEP = '\x01'
TWEET_SEP = '\x02'
TWEETS_TEXT = 'tweets_text.txt'
OPT_TEXT = 'optimized_tweets.txt'
DATASET = 'dataset.pkl.gz'
BLNC = 70

def rmv_punctuation(text : str) -> str:
    return text.replace(';', ' ').replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ').replace('"', ' ').replace(' \n', '\n').replace('\n ', '\n').replace('  ', ' ')


def raw_to_opt(raw_tweets : str, opt_tweets : str):
    '''
    Extragem tweeturile din fisierul original si le\n
    prelucram pentru a putea opera mai usor cu ele
    '''

    # deschidem si extragem tweeturile din fisierul original
    resource_file = open(raw_tweets)
    text = resource_file.read()
    resource_file.close()
    new_text = re.findall('[0-9]+\t[^\t]+\t[01]', text)

    all_set = []

    for i in new_text:
        all_set.append(i.split('\t'))

    # scriem in fisierul opt_tweets separate mai eficient

    all_set_size = len(all_set)
    new_file = open(opt_tweets, 'w')
    for i in range(all_set_size - 1):
        new_file.write(all_set[i][1] + TWEET_SEP + all_set[i][2] + TXT_SEP)
    new_file.write(all_set[all_set_size - 1][1] + TWEET_SEP + all_set[all_set_size - 1][2])
    new_file.close()

    # se verifica integritatea datelor
    print(test_opt(opt_tweets))

    # se arhiveaza datele
    opt_tweets_dump(opt_tweets)



def test_opt(opt_tweets : str) -> bool:
    '''
    Un mic test in care vedem daca outputul\n
    este cel asteptat
    '''
    opt_file = open(opt_tweets)
    text = opt_file.read()
    opt_file.close()
    all_set = text.split(TXT_SEP)

    for i, tweet in enumerate(all_set):
        all_set[i] = tweet.split(TWEET_SEP)
        if all_set[i][1] != '0' and all_set[i][1] != '1':
            return False
    return True

def opt_tweets_dump(opt_tweets : str):
    file_obj = open(opt_tweets)
    opt_text = file_obj.read()
    file_obj.close()

    opt_text = optimize_text(opt_text)

    dataset = opt_text.split(TXT_SEP)

    true_tweets = []
    true_output = []
    false_tweets = []
    false_output = []

    for i, tweet in enumerate(dataset):
        dataset[i] = tweet.split(TWEET_SEP)
        if dataset[i][1] == '1':
            true_tweets.append(dataset[i][0])
            true_output.append(1)
        else:
            false_tweets.append(dataset[i][0])
            false_output.append(0)

    true_set = [true_tweets, true_output]
    false_set = [false_tweets, false_output]

    train_set, test_set = balance_data(true_set, false_set)
    
    file_obj = gzip.open(DATASET, 'wb')
    pickle.dump([train_set, test_set], file_obj)
    print('S-au introdus datele in fisier!')
    file_obj.close()


def optimize_text(tweets : str) -> str:
    ret_text = copy.deepcopy(tweets)

    # remove punctuation
    ret_text = ret_text.replace(',', ' ').replace('.', ' ')
    ret_text = ret_text.replace('?', ' ').replace('!', ' ')
    ret_text = ret_text.replace(';', ' ').replace('"', ' ').replace('“', ' ')
    ret_text = ret_text.replace('\n', ' ').replace('\t', ' ')

    # transform aronds
    ret_text = re.sub(f"@[^ \n]+[ \n\x00{TWEET_SEP}{TXT_SEP}]", ' REF ', ret_text)

    # transform emojies
    emojiz = open("emoji.txt").read().split('\n')
    for i in emojiz:
        i = i.split(' ')
        ret_text = ret_text.replace(i[0], f" {i[1]} ")

    # remove extra chars
    ret_text = ret_text.replace('\x2F', ' ').replace('=', ' ').replace(':', ' ').replace('#', ' ').replace('*', ' ').replace('-', ' ').replace('…', ' ')

    # remove paranthesis
    ret_text = ret_text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')

    # lower text
    ret_text = ret_text.lower()

    # remove extra spaces
    ret_text = ret_text.replace('   ', ' ').replace('  ', ' ')

    # remove first and last spaces
    ret_text = re.sub(f'{TXT_SEP} ', TXT_SEP, ret_text)
    ret_text = re.sub(f' {TWEET_SEP}', TWEET_SEP, ret_text)

    return ret_text

def balance_data(true_set : list, false_set : list) -> tuple:
    a = list(zip(true_set[0], true_set[1]))
    random.shuffle(a)

    b = list(zip(false_set[0], false_set[1]))
    random.shuffle(b)


    train_set = a[:(BLNC * len(a) // 100)] + b[:(BLNC * len(b) // 100)]
    random.shuffle(train_set)
    train_set = list(zip(*train_set))

    test_set = a[(BLNC * len(a) // 100):] + b[(BLNC * len(b) // 100):]
    random.shuffle(test_set)
    test_set = list(zip(*test_set))

    return train_set, test_set


    



if __name__ == "__main__":
    raw_to_opt(TWEETS_TEXT, OPT_TEXT)

