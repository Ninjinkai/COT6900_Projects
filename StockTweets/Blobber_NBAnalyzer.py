import pprint
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')
# input_csv = open('./data/msg-sent-new-no-quotes.csv', 'r')

output_csv = open('./data/NB-blobber-results.csv', 'w')
output_csv.write("tweet,stated_sentiment,blob_NB_classification,blob_NB_p_pos,blob_NB_p_neg\n")

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
    blob_NB_sentiment = blob(tweet).sentiment
    blob_NB_classification = blob_NB_sentiment.classification
    blob_NB_p_pos = blob_NB_sentiment.p_pos
    blob_NB_p_neg = blob_NB_sentiment.p_neg

    output_line = tweet + "," + stated_sentiment + "," \
                  + blob_NB_classification + "," + str(blob_NB_p_pos) + "," + str(blob_NB_p_neg)+ "\n"
    output_csv.write(output_line)

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment,
                    'blob_NB_classification': blob_NB_classification,
                    'blob_NB_p_pos': blob_NB_p_pos,
                    'blob_NB_p_neg': blob_NB_p_neg})

print("Process finished.")
pprint.pprint(results)

input_csv.close()
output_csv.close()