import pprint
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')
# input_csv = open('./data/msg-sent-new-no-quotes.csv', 'r')

output_csv = open('./data/blobber-results.csv', 'w')

print("CSVs opened.")

blob = Blobber(analyzer=NaiveBayesAnalyzer())

print("Analyzer created.")

results = []
counter = 0

print("Analyzing tweets.")

for line in input_csv:

    if counter % 10000 == 0:
        print('.')
    counter += 1

    tweet = line.split(',')[0]
    stated_sentiment = line.split(',')[1].strip()
    blob_sentiment = blob(tweet).sentiment
    blob_classification = blob_sentiment.classification
    blob_p_pos = blob_sentiment.p_pos
    blob_p_neg = blob_sentiment.p_neg

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment,
                    'blob_classification': blob_classification,
                    'blob_p_pos': blob_p_pos,
                    'blob_p_neg': blob_p_neg})

print("Process finished.")
# pprint.pprint(results)

input_csv.close()
output_csv.close()