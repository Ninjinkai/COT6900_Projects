import pprint
from textblob import TextBlob

input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')
# input_csv = open('./data/msg-sent-new-no-quotes.csv', 'r')

output_csv = open('./data/def-blobber-results.csv', 'w')
output_csv.write("tweet,stated_sentiment,blob_polarity,blob_subjectivity\n")

print("CSVs opened.")

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
    blob_sentiment = TextBlob(tweet).sentiment
    blob_polarity = blob_sentiment.polarity
    blob_subjectivity = blob_sentiment.subjectivity

    output_line = tweet + "," + stated_sentiment + ","\
                  + str(blob_polarity) + "," + str(blob_subjectivity) + "\n"
    output_csv.write(output_line)

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment,
                    'blob_polarity': blob_polarity,
                    'blob_subjectivity': blob_subjectivity})

print("Process finished.")
pprint.pprint(results)

input_csv.close()
output_csv.close()