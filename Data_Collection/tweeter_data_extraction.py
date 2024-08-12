
import tweepy
import csv


# Consumer keys and access tokens, used for OAuth
consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXX'



auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Open/Create a file to append data
csvFile = open('20190902/'+'20190902_mtnUG.csv', 'w')

#Use csv Writer
csvWriter = csv.writer(csvFile)

csvWriter.writerow(['created_at', 'tweet_id', 'tweet_id_str', 'tweet_full_text', 'tweet_user_followers_count', 'tweet_user_friends_count', 'tweet_user_listed_count', 'tweet_user_created_at', 'tweet_user_favourites_count', 'tweet_coordinates', 'tweet_place','tweet_retweet_count', 'tweet_favorite_count', 'tweet_user_id', 'tweet_user_id_str', 'tweet_user_name', 'tweet_user_screen_name', 'tweet_user_location', 'tweet_user_description'])

 
## Twitter handles of Telecom companies
#mtnug 

# Define the search term and the date_since date as variables
search_words = "mtnUG"


#Below you ignore all retweets by adding -filter:retweets to your query. 
#The Twitter API documentation has information on other ways to customize your queries.

new_search = search_words + " -filter:retweets"
    
for tweet in tweepy.Cursor(api.search,
                           q=new_search,
                           since="2019-08-27",
                           until="2019-09-02",
                           retweeted='false',
                           lang="en",
                           tweet_mode='extended').items():

    
    #print (tweet.created_at,tweet.id, tweet.id_str,tweet.user.followers_count,tweet.retweet_count, tweet.full_text )
    print(tweet.created_at, tweet.id, tweet.id_str, tweet.full_text, tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count, tweet.user.created_at, tweet.user.favourites_count, tweet.coordinates, tweet.place, tweet.retweet_count, tweet.favorite_count, tweet.user.id, tweet.user.id_str, tweet.user.name, tweet.user.screen_name, tweet.user.location, tweet.user.description)
    csvWriter.writerow([tweet.created_at, tweet.id, tweet.id_str, tweet.full_text.encode('utf-8'), tweet.user.followers_count, tweet.user.friends_count, tweet.user.listed_count, tweet.user.created_at, tweet.user.favourites_count, tweet.coordinates, tweet.place, tweet.retweet_count, tweet.favorite_count, tweet.user.id, tweet.user.id_str, tweet.user.name.encode('utf-8'), tweet.user.screen_name.encode('utf-8'), tweet.user.location.encode('utf-8'),tweet.user.description.encode('utf-8')])
    
csvFile.close()
