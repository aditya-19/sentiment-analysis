# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:08:05 2020

@author: Aditya
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tweepy
import tensorflow as tf
import re
model=tf.keras.models.load_model("my_model2.h5")
amazon_df=pd.read_csv(r"C:\Users\Aditya\Desktop\Time Pass\Project\sentiment labelled sentences\sentiment labelled sentences\amazon_cells_labelled.txt",
                        delimiter='\t',
                        header=None, 
                        names=['review', 'sentiment'])

imdb_df = pd.read_csv(r"C:\Users\Aditya\Desktop\Time Pass\Project\sentiment labelled sentences\sentiment labelled sentences\imdb_labelled.txt", 
                        delimiter='\t', 
                        header=None, 
                        names=['review', 'sentiment'])

yelp_df = pd.read_csv(r"C:\Users\Aditya\Desktop\Time Pass\Project\sentiment labelled sentences\sentiment labelled sentences\yelp_labelled.txt", 
                        delimiter='\t', 
                        header=None, 
                        names=['review', 'sentiment'])

data=pd.concat([amazon_df,yelp_df,imdb_df])
data.reset_index(drop='True',inplace=True)


#Extracting Reviews and Sentiments
sentences=data['review'].tolist()
label=data['sentiment'].tolist()
    
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size=1000
tokenizer=tfds.features.text.SubwordTextEncoder.build_from_corpus(
        sentences,vocab_size,max_subword_length=5)

#Predict Reviews function
max_length=70
def predict_review(model, new_sentences, maxlen=max_length, show_padded_sequence=True ):
    # Keep the original sentences so that we can keep using them later
    # Create an array to hold the encoded sequences
    new_sequences = []

    # Convert the new reviews to sequences
    for i, frvw in enumerate(new_sentences):
        new_sequences.append(tokenizer.encode(frvw))

    trunc_type='post' 
    padding_type='post'

    # Pad all sequences for the new reviews
    new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length, 
                                 padding=padding_type, truncating=trunc_type)             

    classes = model.predict(new_reviews_padded)

    # The closer the class is to 1, the more positive the review is
    pos=0
    neg=0
    neu=0
    for x in range(len(new_sentences)):

        # We can see the padded sequence if desired
        # Print the sequence
        if (show_padded_sequence):
              print(new_reviews_padded[x])
        # Print the review as text
        print(new_sentences[x])
        # Print its predicted class
        #print(classes[x])
        print ('%.2f'%classes[x])
        
        val=float('%.3f'%classes[x])
        print("Val: ",val)
        if(val>=0.80 and val<=1.00):
          pos=pos+1
        elif(val>=0.45 and val<=0.79):
          neu=neu+1
        else:
          neg=neg+1
          
        print("\n")
    print("+tive reviews: ",pos,"\n Neutral Reviews",neu,"\n -tive Reviews",neg) 
    labels = 'Positive Review', 'Negative Review', 'Neutral Review'
    sizes = [pos,neg,neu]
    explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    
    labels_list=["Positive Review","Negative Review","Neutral Review"]
    index = np.arange(len(labels_list))
    plt.bar(index, sizes)
    plt.xlabel('Sentiments', fontsize=15)
    plt.ylabel('No of Reviews', fontsize=15)
    plt.xticks(index, labels_list, fontsize=10, rotation=0)
    plt.title('Sentiment Analysis Distribution of Tweets')
    plt.show() 

#Preprocessing of tweets function
def preprocessing_of_tweets(real_reviews):
    new_real_reviews=[]
    emoji_pattern = re.compile("["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 u"\U00002702-\U000027B0"
                                 u"\U000024C2-\U0001F251"
                                 "]+", flags=re.UNICODE)

    for tweet in real_reviews:
        tweet = re.sub(r"^https://t.co/[A-Za-z0-9]*\s"," ",tweet)
        tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
        tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
        tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F])+)'," ",tweet)
        tweet = re.sub(r'^[a-zA-Z0-9]?.co/[a-zA-Z0-9]*'," ",tweet)
                                             # to substitute url's with a whitespace at the beginning, in between and at the end of a sentence
        tweet = re.sub("\.\.+"," ",tweet)    # to substitute more than one '.' between words with a space
        tweet = re.sub("-$","",tweet)        # to remove '-' at the end of a sentence
        tweet = re.sub(r"^ +","",tweet)      # to remove one or more whitespace at the beginning of a sentence
        tweet = re.sub(r"  +"," ",tweet)     # to substitute one or more whitespace with a single space
        tweet = emoji_pattern.sub(r'', tweet)

        new_real_reviews.append(tweet)
        
    return new_real_reviews


#Bring me tweets
def bring_me_tweets(keyword):
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler("88SjTJ8H5TS2MC5XGXrnois8T" , "j1uj0gjnND8Y3HEDTe5o7gUEnUvPu4p1grlVqeTIVAiyT40jyI")
    auth.set_access_token("1256913946298273792-0PusHLHqJdUt4Cv5S08TCNdwp8L3HW", "XYiEXGXZCcRnQAhofPPuCdRmNAEaQRZI8YYPu9cMNC07m")
    # Create API object
    api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
    if(not api):
        print("Can't Authenticate")
    else:
        print("Done with Authentication")
    
    #Lets look for tweets aboout the keyword
    searchQuery = keyword #@param {type:"string"}
    maxTweets = 100
    Filter_Retweets = True #@param {type:"boolean"}
    tweetsPerQry = 100  # this is the max the API permits
    tweet_lst = []
    if Filter_Retweets:
        searchQuery = searchQuery + ' -filter:retweets'  # to exclude retweets
    sinceId = None
    #max_id = -10000000000
    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            if(not sinceId):
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en")
            else:
                new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                        lang="en", since_id=sinceId)
            if not new_tweets:
                print("No tweets on this topic ...")
                break
            else:
                print("Length new_tweets list: ",len(new_tweets))
            for tweet in new_tweets:
                if hasattr(tweet,'reply_count'):
                    reply_count=tweet.reply_count
                else:
                    reply_count=0
                if hasattr(tweet,'retweeted'):
                    retweeted=tweet.retweeted
                else:
                    retweeted='NA'
                #fixup search query to get topic
                topic = searchQuery[:searchQuery.find('-')].capitalize().strip()
                # fixup date
                tweetDate = tweet.created_at.date() 
            
                tweet_lst.append([tweetDate, topic,
                              tweet.id, tweet.user.screen_name, tweet.user.name,
                              tweet.text, tweet.favorite_count,
                              reply_count, tweet.retweet_count, retweeted])
            
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            #max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("Inside except block")
            print("some error : " + str(e))
            break
    print("Downloaded {0} tweets".format(tweetCount))
    print("Finally tweet_lst: ",len(tweet_lst))

    return tweet_lst
        
keyword="iPhone"
real_reviews = bring_me_tweets(keyword)
        
pd.set_option('display.max_colwidth', -1)
# load it into a pandas dataframe
tweet_df = pd.DataFrame(real_reviews, columns=['tweet_dt', 'topic', 'id', 'username', 'name',
                                                    'tweet', 'like_count', 'reply_count',
                                                    'retweet_count', 'retweeted'])
tweet_df.to_csv('tweets.csv') 
tweet_df.head()
#finally some realtime reviews
some_reviews=tweet_df['tweet'].tolist()
print("Length of real_reviews list : ",len(some_reviews))
final_reviews = preprocessing_of_tweets(some_reviews)
print("Length of final_reviews list : ",len(final_reviews))

predict_review(model,final_reviews)
