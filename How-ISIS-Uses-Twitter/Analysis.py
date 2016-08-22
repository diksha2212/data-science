### Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

##LOADING DATA

tweets=pd.read_csv("tweets.csv")

#### MOST COMMONLY USED KEYWORDS ####
### "baqiyah" --state, "dabiq" --ISIS magazine, "wilayat" --authority, "amaq" --ISIS media outlet #####


baqiyah_instances = tweets.tweets.str.contains(r'baqiyah').sum() 
dabiq_instances = tweets.tweets.str.contains(r'dabiq').sum() 
wilayat_instances = tweets.tweets.str.contains(r'wilayat').sum() 
amaq_instances = tweets.tweets.str.contains(r'amaq').sum() 

##PLOTTING OCCURENCES OF VARIOUS INSTANCES ##

X=("Baqiyah","Dabiq","Wilayat","Amaq")
Y=(baqiyah_instances,dabiq_instances,wilayat_instances,amaq_instances)

ax = sns.barplot(x=X, y=Y)
plt.show()

## AMAQ is the most commonly used keyword among the Pro ISIS fanboys

### DATA CATEGORIZATION OF LINKS ###

audio_instances = tweets.tweets.str.contains(r'audio').sum() 
video_instances = tweets.tweets.str.contains(r'video').sum()
image_instances = tweets.tweets.str.contains(r'image').sum() + tweets.tweets.str.contains(r'photo').sum()
jihad_website_instances = tweets.tweets.str.contains(r'http').sum() + tweets.tweets.str.contains(r'https').sum() + tweets.tweets.str.contains(r'link').sum()
mainstream_media_instances = tweets.tweets.str.contains(r'TV').sum() + tweets.tweets.str.contains(r'radio').sum()


##PLOTTING OCCURENCES OF VARIOUS INSTANCES ##

X=("AUDIO","VIDEO","IMAGE UPLOADS","JIHADIST WEBSITES","TV/RADIO")
Y=(audio_instances,video_instances,image_instances,jihad_website_instances,mainstream_media_instances)

ax = sns.barplot(x=X, y=Y)
plt.show()

### VISUALIZING ALL THE TWEETS OVER A TIMELINE ###

tweets.time=pd.to_datetime(tweets.time)

%matplotlib inline

tweets.time.value_counts().plot(title='ISIS tweets over time',xlim=(min(tweets.time),max(tweets.time)),figsize=(16,8))

#### tweets in april 2016###

tweets.time.value_counts().plot(title='ISIS tweets in april 2016',xlim=('2016-4-1','2016-4-30'),figsize=(16,8))

#### tweets in Feb 2016###

tweets.time.value_counts().plot(title='ISIS tweets in april 2016',xlim=('2016-2-1','2016-2-28'),figsize=(16,8))

###no. of tweets vs name####

top10_tweeters=pd.DataFrame(tweets['name'].value_counts().head(n=10)).reset_index()
top10_tweeters

##Analysing no of tweets vs no of followers for top 10 tweeters 
top_10_tweeters = ("Rami","War BreakingNews",'Conflict Reporter','Salahuddin Ayubi','Ibni Haneefah','wayf44rer',"كتكات كوكونة",'N i d a l','راعي البقر جوز الهند','Maghrabi Arabie')

followers_of_top10 = []
for name in top_10_tweeters:
    followers_of_top10.append(tweets[tweets['name']==name].head(n=1)['followers']) 


### Testing whether name matches username ##
dict={}
for name in tweets['name']:
    dict[name]= tweets[tweets['name'] == name]['username'].unique()
    
for key in dict:
    values = dict[key]
    if(len(values))>1:
        print(key,values)
        
### there are multiple fanboys who use differnet usernames###

###SENTIMENT ANALYSIS OF TWEETS ###

### TEXT CLEANING OF TWEETS ##
import nltk
from nltk.corpus import stopwords

def preprocess_tweet( tweet ):
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", tweet) 

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
  
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    return( " ".join( meaningful_words ))   

tweets['tweets']=tweets['tweets'].apply(preprocess_tweet)

##3splitting into train and test ###
from sklearn.cross_validation import train_test_split

train_data, test_data = train_test_split(tweets, test_size = 0.2)

##BUILDING WORD COUNT VECTOR FOR EACH REVIEW

from sklearn.feature_extraction.text import CountVectorizer

#vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')#
vectorizer=CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000) 
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['tweets'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['tweets'])

## ASSigning Initial labels on basis of clergy ###

Pro_ISIS_Clergy = ("abdul","wahhab","anwar","awlaki","ahmad","jibrill","ibn","taymiyyah")
Anti_ISIS_Clergy =("nouman ali khan", "yaqoubi","hamza","yusuf","suhaib","webb","yaser","qadhi","nouman","ali khan","khan")


def assign_initial_sentiment(tweet):
    negative_score=0
    positive_score=0

    if any(word in tweet for word in Pro_ISIS_Clergy):
         positive_score+=1
    if any(word in tweet for word in Anti_ISIS_Clergy):
         negative_score+=1
               
    sentiment = 0
    if(positive_score> negative_score):
        sentiment = 1
    elif(negative_score>positive_score):
        sentiment =-1
    return sentiment
    
  
  #### Assigning labels to train data ###

train_data['Sentiment']=train_data.tweets.apply(assign_initial_sentiment)
test_data['Sentiment']=test_data.tweets.apply(assign_initial_sentiment)

###Predicting sentiment for test data ##
from sklearn import linear_model

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(train_matrix, train_data['Sentiment'])


 scores = logreg.decision_function(test_matrix)
predictions=logreg.predict(test_matrix)
probability_predictions=logreg.predict_proba(test_matrix)
