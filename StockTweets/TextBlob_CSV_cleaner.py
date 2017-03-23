input_csv = open('./data/msg-sent-01-short.csv', 'r')
output_csv = open('./data/msg-sent-01-short-no-quotes.csv', 'w')


for line in input_csv:
    output_csv.write(line.replace('\'',''))

input_csv.close()
output_csv.close()