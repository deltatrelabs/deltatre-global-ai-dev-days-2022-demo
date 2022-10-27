# Twitter API

Having at disposals tweets or generic posts on social media can be incredible source of information. It is possible to do reputation analysis, sentiment analysis on a particular topic or train bot/AI to understand and generate content.

Twitter in particular has its own API that can be used by developers who need to download tweets, manage stream of contents or study users. Naturally you need credentials, and credentials are obtained if you get a developer account.

## Get developer account

Naturally you need a twitter account and request for a [developer account](https://developer.twitter.com/en/portal/dashboard): sign up, insert you anagraphical information and register.

Now you should have access to the `Twitter Developer Platform` where yuo can build your first app: choose a name, get the keys and save them (I copied and pasted in a .txt file). __Keep the keys secret and save them because later you won't have access any more__ (you will be able to generate other ones).

Go to the `Dashboard`: generate, see and save the access token and the secret access token.

Now you have an `Essential` account with some limitations, if you want to increase your capabilities (i.e.: increase the number of tweets you can download).

To have an elevated account you need to go to `Products` tend and choose the `Elevated` tab and click on `Apply`. Twitter will ask you few things about you, your project and your scope, answer and be precise because if the information are not clear they can reject your request. In few hours (it took me minutes actually) Twitter will reply to your request.

Now you have an `Elevated Twitter Developer Account`, you can start interacting with tweets and account through the API. If anything is not clear I followed this [video](https://www.youtube.com/watch?v=Lu1nskBkPJU&t=4s&ab_channel=AISpectrum).


## Get tweets

To interact with Twitter using Python, you can use [Tweepy](https://docs.tweepy.org/en/stable/) which is a python library (you can install it using pip).

First of all I created a `config.ini` file where I stored the keys and the access tokens. Through `configparser` it is possible to import these sensible information without copying and pasting them in the code (avoid sharing them in public).

Once imported, you can use the `tweepy.OAuthHandler(api_key, api_key_secret)` object to authenticate.

```python

config = configparser.ConfigParser()
config.read('config.ini')

api_key=config['twitter']['api_key']
api_key_secret=config['twitter']['api_key_secret']

acces_token=config['twitter']['acces_token']
access_token_secret=config['twitter']['access_token_secret']

# authentication

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(acces_token, access_token_secret)
api = tweepy.API(auth)
```

Now you can download users tweet using the `user_timeline` method, but this has some limits (100 tweets at max). You can overcome these limitations using the `Cursor` object that allows you to repeat the operation multiple time.

The tweets objects have many features: text, timestamp, user, status, location... I decided to download tweets, save some useful features and store them in a `pd.DataFrame` to export as a CSV file.

```python
user = 'user_name'

tweets = tweepy.Cursor(api.user_timeline, screen_name=user, count=200, tweet_mode='extended').items(limit)
columns = ['Time', 'Text', 'Entities', 'In_reply_to_status_id', 'Language']
data = []

for tweet in tweets:
    text = str( tweet.full_text.encode('ascii',errors='ignore'))
    data.append([tweet.created_at , text, tweet.entities, tweet.in_reply_to_status_id, tweet.lang])

df = pd.DataFrame(data, columns=columns)
df.to_csv(f'{output_name}.csv', sep=',')
```
