from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

input_csv = open('./data/BodySent-new.csv', 'r')

output_csv = open('./data/vader-new-01.csv', 'w')

output_csv.write("sentiment,label\n")

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

    if vader_analysis['pos'] > vader_analysis['neg']:
        output_string = "1," + str(stated_sentiment) + "\n"
    elif vader_analysis['neg'] > vader_analysis['pos']:
        output_string = "0," + str(stated_sentiment) + "\n"
    else:
        output_string = "999," + str(stated_sentiment) + "\n"

    output_csv.write(output_string)

print("Process finished.")

input_csv.close()
output_csv.close()