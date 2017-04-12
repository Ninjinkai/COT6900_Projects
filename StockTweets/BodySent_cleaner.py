
input_csv = open('./data/BodySent.csv', 'r')
output_csv = open('./data/BodySent-new.csv', 'w')

for line in input_csv:
    tweet = line.split(',')[0].strip()
    while '"' in tweet:
        tweet = tweet.replace('"', '')
    while '  ' in tweet:
        tweet = tweet.replace('  ', ' ')
    sentiment = line.split(',')[1].strip()

    output_csv.write(tweet + ',' + sentiment + '\n')

input_csv.close()
output_csv.close()