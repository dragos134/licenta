import gensim.downloader as api

if __name__ == "__main__":
    wv = api.load('word2vec-google-news-300')
    print(wv['king'])
