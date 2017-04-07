import pprint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')
# input_csv = open('./data/msg-sent-new-no-quotes.csv', 'r')

output_csv = open('./data/vader-results.csv', 'w')

print("CSVs opened.")

vaderAnalyzer = SentimentIntensityAnalyzer()

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
    vader_analysis = vaderAnalyzer.polarity_scores(tweet)

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment,
                    'vader_analysis_compound': vader_analysis['compound'],
                    'vader_analysis_neg': vader_analysis['neg'],
                    'vader_analysis_neu': vader_analysis['neu'],
                    'vader_analysis_pos': vader_analysis['pos']})

print("Process finished.")
# pprint.pprint(results)

input_csv.close()
output_csv.close()