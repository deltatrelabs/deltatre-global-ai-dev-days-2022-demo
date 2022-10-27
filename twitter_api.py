import tweepy
import configparser
import pandas as pd
import click

@click.command()
@click.option('--limit', type=click.INT, required=True, help='Limit to tweets to download', default=5000)
@click.option('--output_name', type=click.STRING, required=True, help='name of the output csv')
@click.option('--twitter_username', type=click.STRING, required=True, help='twitter account user name')
def main(limit, output_name, twitter_username):

    # read credential from config file

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

    # get tweets from account
    user = twitter_username

    tweets = tweepy.Cursor(api.user_timeline, screen_name=user, count=200, tweet_mode='extended').items(limit)
    columns = ['Time', 'Text', 'Entities', 'In_reply_to_status_id', 'Language']
    data = []

    for tweet in tweets:
        text = str( tweet.full_text.encode('ascii',errors='ignore'))
        data.append([tweet.created_at , text, tweet.entities, tweet.in_reply_to_status_id, tweet.lang])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'{output_name}.csv', sep=',')

    print(df)

    # Useful tutorials
    # https://www.youtube.com/watch?v=Lu1nskBkPJU&ab_channel=AISpectrum
    # https://www.youtube.com/watch?v=FmbEhKSpR7M&ab_channel=AISpectrum

if __name__ == "__main__":
    main()