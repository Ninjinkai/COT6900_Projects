import pprint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

# input_csv = open('./data/msg-sent-short-no-quotes.csv', 'r')
# input_csv = open('./data/msg-sent-new-no-quotes.csv', 'r')
input_csv = open('./data/BodySent-new.csv', 'r')

# output_csv = open('./data/vader-results.csv', 'w')
output_csv = open('./data/vader-results-BodySent-full.csv', 'w')

output_csv.write(
    "tweet,stated_sentiment,vader_analysis_compound,vader_analysis_neg,vader_analysis_neu,vader_analysis_pos\n")

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

    output_line = tweet + "," + stated_sentiment + "," \
                  + str(vader_analysis['compound']) + "," + str(vader_analysis['neg']) + ","\
                  + str(vader_analysis['neu']) + "," + str(vader_analysis['pos']) + "\n"
    output_csv.write(output_line)

    results.append({'tweet': tweet, 'stated_sentiment': stated_sentiment,
                    'vader_analysis_compound': vader_analysis['compound'],
                    'vader_analysis_neg': vader_analysis['neg'],
                    'vader_analysis_neu': vader_analysis['neu'],
                    'vader_analysis_pos': vader_analysis['pos']})

print("Process finished.")
# pprint.pprint(results)

vader_pickle_file = open('./pickles/vader.pickle', 'wb')
pickle.dump(results, vader_pickle_file)
vader_pickle_file.close()

input_csv.close()
output_csv.close()