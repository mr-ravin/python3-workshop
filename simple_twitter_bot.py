import random
import tweepy

consumer_key="type_your_key_here"
consumer_secret="type_your_key_secret_here"
access_token="type_your_token_here"
access_token_secret="type_your_token_secret_here"

filename="quote.txt" # a text file which contains quotes (one quote in each line)

def file_read(filename):        # this function takes file name as input and returns list of string 
    data_link=open(filename,"r")
    dt = data_link.readlines()
    for idx in range(len(dt)):
        dt[idx] = dt[idx][:-1]  # this is used to remove \n
    return dt

def select_tweet(list_tweet):  # this function takes list of strings as input and then return a randomly selected element of input list
    sz = len(list_tweet)
    idx = random.randint(0,sz-1)  # sz-1 is used because index of list starts from 0 and goes to N-1 for N elements
    msg = list_tweet[idx]
    return msg

def start(consumer_key,consumer_secret,access_token,access_token_secret,filename="quote.txt"): 
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)                                   
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True,compression=True)
    file_data = file_read(filename)
    msg=select_tweet(file_data)
    api.update_status(msg)

start(consumer_key,consumer_secret,access_token,access_token_secret,filename)
