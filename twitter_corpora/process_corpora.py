import time
import re


def process_t1():
    print('se proceseaza fisierul tweets1.txt .....')
    f = open(f"./twitter_corpora/tweets1.txt")
    file_contents = f.read().split('\n')
    f.close()

    f = open('./twitter_corpora/new_tweets1.txt', 'w')
    for i in file_contents:

        i = i.split('\t')
        if len(i) == 4:
            text = i[2]
            text = re.sub(r'[^a-zA-Z\']', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = text.lower()
            f.write(text + '\n')
    f.close()

def process_t2():
    print('se proceseaza fisierul tweets2.txt .....')
    f = open(f"./twitter_corpora/tweets2.txt")
    file_contents = f.read().split('\n')
    f.close()

    f = open(f"./twitter_corpora/new_tweets2.txt", 'w')
    for i in file_contents:
        i = i.split('","')
        if len(i) == 6:
            text = i[5]
            text = re.sub(r'[^a-zA-Z\']', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = text.lower()
            f.write(text + '\n')
    f.close()

def process_t3():
    print('se proceseaza fisierul tweets3.txt .....')
    f = open(f"./twitter_corpora/tweets3.txt")
    file_contents = f.read().split('\n')
    f.close()

    f = open(f"./twitter_corpora/new_tweets3.txt", 'w')
    for i in file_contents:
        i = i.split('\t')
        if len(i) == 4:
            text = i[2]
            text = re.sub(r'[^a-zA-Z\']', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = text.lower()
            f.write(text + '\n')
    f.close()

def combine_corpora():
    print('se combina fisierele....')
    f = open('./twitter_corpora/corpora.txt', 'w')
    for i in range(3):
        new_f = open(f'./twitter_corpora/new_tweets{i+1}.txt', 'r')
        f.write(new_f.read())
        new_f.close()
    f.close()

if __name__ == "__main__":
    process_t1()
    process_t2()
    process_t3()
    combine_corpora()