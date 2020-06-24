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
TWEETS_TEXT = './dataset/tweets_text.txt'
OPT_TEXT = './dataset/optimized_tweets.txt'
DATASET = './dataset/dataset.pkl.gz'
BLNC = 70
SHORTS = './preprocessing/shorts.txt'
USELESS = './preprocessing/useless_words.txt'
EMOJI = './preprocessing/emoji.txt'
MEDICINE = './preprocessing/medicine'


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

    print(f'Am gasit {len(new_text)} tweeturi!')

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
    print(f'Avem {len(train_set[0])+len(test_set[0])} tweeturi')
    file_obj.close()


def optimize_text(tweets : str) -> str:
    ret_text = copy.deepcopy(tweets)

    # lower text
    ret_text = ret_text.lower()

    # replace adr's
    print('incepe preprocesarea ADR')
    start_time = time.time()
    f = open('./preprocessing/ADR_lexicon.tsv', 'r')
    adr_lex = f.read().split('###############################################################')[2].split('\n')
    for line in adr_lex:
        if len(line) > 0:
            line = line.split('\t')
            ret_text = ret_text.replace(line[1], ' ADR ')
    f.close()

    print("Inlocuirea efectelor adverse s-a terminat in %.2f secunde" % (time.time() - start_time))

    # replace drug words
    print('incepe preprocesarea medicamentelor')
    start_time = time.time()
    f = open('./preprocessing/drug_names.txt')

    drug_names = f.read().split('\n')

    for line in drug_names:
        ret_text = ret_text.replace(line, ' MED ')

    f.close()
    print("Inlocuirea medicamentelor s-a terminat in %.2f secunde" % (time.time() - start_time))

    #replace "'" words
    f = open(SHORTS, 'r')
    text = f.read().split('\n')
    for word in text:
        word = word.split(',')
        ret_text = ret_text.replace(word[0], word[1])
    f.close()

    # transform links
    ret_text = re.sub(r"http[^\s\x02\x01]*", ' LINK ', ret_text)
    
    # remove punctuation
    ret_text = ret_text.replace(',', ' ').replace('.', ' ').replace("\x27", ' ')
    ret_text = ret_text.replace('?', ' ').replace('!', ' ')
    ret_text = ret_text.replace(';', ' ').replace('"', ' ').replace('“', ' ').replace('”', ' ')
    ret_text = ret_text.replace('\n', ' ').replace('\t', ' ')

    # transform aronds
    ret_text = re.sub(f"@[^ \n\x00{TWEET_SEP}{TXT_SEP}]+", ' REF ', ret_text)

    # transform emojies
    emojiz = open(EMOJI).read().split('\n')
    for i in emojiz:
        i = i.split(' ')
        ret_text = ret_text.replace(i[0], f" {i[1]} ")

    # remove extra chars
    ret_text = ret_text.replace('\x2F', ' ').replace('=', ' ').replace(':', ' ').replace('#', ' ').replace('*', ' ').replace('-', ' ').replace('…', ' ')

    # remove paranthesis
    ret_text = ret_text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')

    # remove extra spaces
    ret_text = ret_text.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ')

    # remove first and last spaces
    ret_text = re.sub(f'{TXT_SEP} ', TXT_SEP, ret_text)
    ret_text = re.sub(f' {TWEET_SEP}', TWEET_SEP, ret_text)
    return ret_text

def balance_training_data(train : list, ratio : int) -> list:
    true_set = [[], []]
    false_set = [[], []]

    for i in range(len(train[0])):                                      #separam datele true de cele false
        if train[1][i] == 1:
            true_set[0].append(train[0][i])
            true_set[1].append(train[1][i])
        else:
            false_set[0].append(train[0][i])
            false_set[1].append(train[1][i])
    
    ret_set = copy.deepcopy(true_set)

    index_list = []
    for i in range(ratio * len(true_set[0])):                           # adaugam datele false
        new_entry = random.randint(0, len(false_set[0]) - 1)
        while new_entry in index_list:
            new_entry = random.randint(0, len(false_set[0]) - 1)
        ret_set[0].append(false_set[0][new_entry])
        false_set[0].pop(new_entry)
        ret_set[1].append(false_set[1][new_entry])
        false_set[1].pop(new_entry)
        index_list.append(new_entry)

    ret_set = list(zip(ret_set[0], ret_set[1]))
    random.shuffle(ret_set)
    return list(zip(*ret_set)), false_set
    

def balance_data(true_set : list, false_set : list) -> tuple:
    a = list(zip(true_set[0], true_set[1]))
    random.shuffle(a)

    print("Avem %s date cu 1" % (len(a)))

    b = list(zip(false_set[0], false_set[1]))
    random.shuffle(b)

    print("Avem %s date cu 0" % (len(b)))


    train_set = a[:(BLNC * len(a) // 100)] + b[:(BLNC * len(b) // 100)]
    random.shuffle(train_set)
    train_set = list(zip(*train_set))

    test_set = a[(BLNC * len(a) // 100):] + b[(BLNC * len(b) // 100):]
    random.shuffle(test_set)
    test_set = list(zip(*test_set))

    for i in range(2, 6):
        demo_train_set, demo_false = balance_training_data(train_set, i)
        demo_test = copy.deepcopy(test_set)
        demo_test[0] = list(demo_test[0])
        demo_test[1] = list(demo_test[1])
        demo_test[0] = demo_test[0] + demo_false[0]
        demo_test[1] = demo_test[1] + demo_false[1]
        file_obj = gzip.open(f'./dataset/1{i}dataset.pkl.gz', 'wb')
        print(len(demo_train_set[1]), len(demo_test[1]))
        pickle.dump([demo_train_set, demo_test], file_obj)
        file_obj.close()
    

    return train_set, test_set


def write_shorts(dataset: list):
    print('incepe functia write shorts')
    f = open(SHORTS, 'w')
    word_vec = []
    for tweet in dataset:
        tweet = (tweet.split(TWEET_SEP))[0].split(' ')
        # print(tweet)
        # time.sleep(1)
        for word in tweet:
            if '\'' in word and word not in word_vec:
                word_vec.append(word)
                f.write(f'{word}\n')
    f.close()
    exit()




    



if __name__ == "__main__":
    raw_to_opt(TWEETS_TEXT, OPT_TEXT)

