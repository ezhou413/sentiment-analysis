import boto3
import pandas as pd

df = pd.read_csv('data/cao_data.csv')
df['date'] = pd.to_datetime(df.get('date'))

comprehend = boto3.client(service_name='comprehend', region_name='us-west-2')

def get_response(text):
    #return json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True, indent=4)['Sentiment']
    response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return response

df = df.assign(
   response = df.get('description').apply(get_response)
)

def get_sentiment(text):
    return text['Sentiment']
    
def get_sentiment_score(text):
    return text['SentimentScore']

df = df.assign(
    sentiment = df.get('response').apply(get_sentiment),
    sentiment_score = df.get('response').apply(get_sentiment_score)
)

df.to_csv('data/processed/cao.csv')