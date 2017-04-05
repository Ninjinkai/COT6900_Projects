import pprint
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')
# input_csv = open('./data/msg-sent-new-no-quotes.csv', 'r')

print("CSV opened.")

blob = Blobber(analyzer=NaiveBayesAnalyzer())

vaderAnalyzer = SentimentIntensityAnalyzer()

print("Analyzer created.")

results = []
analysis = dict(pos_bear=0, pos_bull=0, neg_bear=0, neg_bull=0)
counter = 0

print("Analyzing tweets", end='')

for line in input_csv:

    if counter % 10000 == 0:
        print('.', end='')
    counter += 1

    tweet = line.split(',')[0]
    stated_sentiment = line.split(',')[1].strip()
    determined_sentiment = blob(tweet).sentiment

    pprint.pprint(vaderAnalyzer.polarity_scores(tweet))

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment, 'determined_sentiment': determined_sentiment})

# pprint.pprint(results)

counter = 0

print("\nCounting results", end='')

for item in results:

    # pprint.pprint(item)
    # print(item['stated_sentiment'], item['determined_sentiment'].classification)

    if counter % 10000 == 0:
        print('.', end='')
    counter += 1

    if item['stated_sentiment'] == 'Bearish' and item['determined_sentiment'].classification == 'pos':
        analysis['pos_bear'] = analysis['pos_bear'] + 1
    elif item['stated_sentiment'] == 'Bullish' and item['determined_sentiment'].classification == 'pos':
        analysis['pos_bull'] = analysis['pos_bull'] + 1
    elif item['stated_sentiment'] == 'Bearish' and item['determined_sentiment'].classification == 'neg':
        analysis['neg_bear'] = analysis['neg_bear'] + 1
    else:
        analysis['neg_bull'] = analysis['neg_bull'] + 1

print('\n')
pprint.pprint(analysis)

input_csv.close()