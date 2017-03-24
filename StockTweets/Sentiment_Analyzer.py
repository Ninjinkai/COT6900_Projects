from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import pprint

input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')

blob = Blobber(analyzer=NaiveBayesAnalyzer())

results = []

for line in input_csv:
    tweet = line.split(',')[0]
    stated_sentiment = line.split(',')[1]
    determined_sentiment = blob(tweet).sentiment

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment, 'determined_sentiment': determined_sentiment})

pprint.pprint(results)

input_csv.close()