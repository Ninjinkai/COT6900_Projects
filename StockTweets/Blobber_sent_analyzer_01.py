from textblob import TextBlob

input_csv = open('./data/BodySent-new.csv', 'r')

output_csv = open('./data/blobber-sent-new-01.csv', 'w')

output_csv.write("sentiment,label\n")

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

    if blob_polarity > 0:
        output_string = "1," + str(stated_sentiment) + "\n"
    elif blob_polarity < 0:
        output_string = "0," + str(stated_sentiment) + "\n"
    else:
        output_string = "999," + str(stated_sentiment) + "\n"

    output_csv.write(output_string)

print("Process finished.")

input_csv.close()
output_csv.close()