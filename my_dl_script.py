import sys, urllib.request, re, json, socket, string
from bs4 import BeautifulSoup as bs
import time

socket.setdefaulttimeout(20)
item_dict = {}

id_file = open('tweets_id.txt', 'r')
new_id_file = open('new_tweets_id.txt', 'w')
text_file = open('tweets_text.txt', 'w')

count = 0
for line in id_file:
    line = line.split('\t')
    tweet_id = line[0]
    user_id = line[1]
    tweet_text = None
    url = 'https://twitter.com/'+str(user_id)+'/status/'+str(tweet_id)
    try:
        tweet_webpage = urllib.request.urlopen(url)
        html = tweet_webpage.read()
        soup = bs(html, features="html.parser")
        jstt = soup.find_all("p", "js-tweet-text")
        tweets = list(set([x.get_text() for x in jstt]))

        if len(tweets) > 1:
            other_tweets = []
            cont = soup.find_all("div", "content")
            for i in cont:
                o_t = i.find_all("p", "js-tweet-text")
                other_text = list(set([x.get_text() for x in o_t]))
                other_tweets += other_text
            tweets = list(set(tweets)-set(other_tweets))
        tweet_text = tweets[0]

        final_form = tweet_id + '\t' + tweet_text + '\t' + line[2]
        text_file.write(final_form)

    except Exception:
        print('tweet not found')
        continue

id_file.close()
text_file.close()
new_id_file.close()
