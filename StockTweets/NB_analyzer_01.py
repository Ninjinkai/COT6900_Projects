from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

input_csv = open('./data/BodySent-new.csv', 'r')

output_csv = open('./data/nb-new-01.csv', 'w')

output_csv.write("sentiment,label\n")

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

    if blob_NB_p_pos > blob_NB_p_neg:
        output_string = "1," + str(stated_sentiment) + "\n"
    elif blob_NB_p_neg > blob_NB_p_pos:
        output_string = "0," + str(stated_sentiment) + "\n"
    else:
        output_string = "999," + str(stated_sentiment) + "\n"

    output_csv.write(output_string)

print("Process finished.")

input_csv.close()
output_csv.close()