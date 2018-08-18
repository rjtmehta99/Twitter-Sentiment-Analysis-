#TODO:
#1. Add negation algorithm for not words
#2. Try to add more and more pre-processing

import tweepy
#from textblob import TextBlob
import matplotlib.pyplot as plt
import pickle
import re
import numpy as np
from nltk.corpus import wordnet
import nltk
import time

#Authenticate
consumer_key = 'kcpINmZC4lRvgTDMH66qMXI9m'
consumer_secret = '1qwyGBwDrddQ9q1UxU802X9W65vjWYfs6b4AUVQUMsJI8sfLfD'

access_token = '1002482290495193088-7dpYGQxJYm3BuYeI1xovh3kOKcnT0O'
access_token_secret = 'BImiX5INiTSD0fWLI0QqXHegjD7vn94sRry42Sd6klfsl'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, timeout= 15)
 
#Retrieve tweets
try:
    public_tweets = api.search(lang = 'en',q = "Dell" , count = 100,result_type = 'recent')
    
except:
    time.sleep(5)    
    public_tweets = api.search(lang = 'en',q = "Dell", count = 100,result_type = 'recent')

with open('tfidfmodel.pickle','rb') as f:
    vectorizer = pickle.load(f)
    
with open('classifier.pickle','rb') as f:
    classifier = pickle.load(f)

total_positive = 0
total_negative = 0

#print("[*] Retreiving Tweets")
#Preprocessing the tweets:
for tweet in public_tweets:
    tweet = tweet.text
    words = nltk.word_tokenize(tweet)
    new_words = []
    temp_word = ""
    for word in words:
        antonyms = []
        if word == "not":
            temp_word = "not_"
        elif temp_word == "not_":
            for syn in wordnet.synsets(word):
                for s in syn.lemmas():
                    for a in s.antonyms():
                        antonyms.append(a.name())
            if len(antonyms) >= 1:
                word = antonyms[0]
            else:
                word = temp_word + word
            temp_word = ""
        
        if word != "not":
            new_words.append(word)
        
    tweet = ' '.join(new_words) 
    
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"they're", "they are",tweet)
    tweet = re.sub(r"we're", "we are",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"isn't","is not",tweet)
    tweet = re.sub(r"aren't","are not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"shan't","shall not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"hasn't","has not",tweet)
    tweet = re.sub(r"hadn't","had not",tweet)
    tweet = re.sub(r"don't","do not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"didn't","did not",tweet)
    tweet = re.sub(r"aint","am not",tweet)
    tweet = re.sub(r"isnt","is not",tweet)
    
    
    tweet = re.sub(r"whats","what is",tweet)
    tweet = re.sub(r"thats","that is",tweet)
    tweet = re.sub(r"theres","there is",tweet)
    tweet = re.sub(r"wheres","where is",tweet)
    tweet = re.sub(r"whos","who is",tweet)
    tweet = re.sub(r"its","it is",tweet)
    tweet = re.sub(r"im","i am",tweet)
    tweet = re.sub(r"theyre", "they are",tweet)
    tweet = re.sub(r"wouldnt","would not",tweet)
    tweet = re.sub(r"aint","am not",tweet)
    tweet = re.sub(r"isnt","is not",tweet)
    tweet = re.sub(r"arent","are not",tweet)
    tweet = re.sub(r"cant","can not",tweet)
    tweet = re.sub(r"shouldnt","should not",tweet)
    tweet = re.sub(r"shant","shall not",tweet)
    tweet = re.sub(r"wont","will not",tweet)
    tweet = re.sub(r"hasnt","has not",tweet)
    tweet = re.sub(r"hadnt","had not",tweet)
    tweet = re.sub(r"dont","do not",tweet)
    tweet = re.sub(r"couldnt","could not",tweet)
    tweet = re.sub(r"didnt","did not",tweet)
    tweet = re.sub(r"aint","am not",tweet)
    tweet = re.sub(r"isnt","is not",tweet)
    
    
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s[a-z]\$"," ",tweet)
    tweet = re.sub(r"\s[a-z]\s"," ",tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"\s+"," ",tweet)
    #print("Tweet \n"+tweet)
    sentiment = classifier.predict(vectorizer.transform([tweet]).toarray())
    if sentiment[0] == 1:
        total_positive += 1
    else:
        total_negative += 1
        
#Visualization
objects = ['Positive','Negative']        
y_pos = np.arange(len(objects))

plt.bar(y_pos,[total_positive,total_negative],alpha = .5)
plt.xticks(y_pos,objects)
plt.ylabel("Number of tweets")
plt.title("Twitter Sentiment Analysis of Dell")

plt.show()