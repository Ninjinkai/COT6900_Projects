import pandas as pd
import numpy as np
from textblob import TextBlob
import csv

#sentiment = []
title = ["sentiment","label"]
with open('./data/Sahar-results.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(title)

DataSet = pd.read_csv('./data/BodySent.csv',header=None, names=['body','sentiment'])
DataSet =  DataSet[DataSet['body'].notnull()]
DataSet =  DataSet[DataSet['sentiment']!=2]

train = DataSet['body'].tolist()
senti = DataSet['sentiment'].tolist()
train.pop(0)
senti.pop(0)

for msg, clas in zip(train, senti):
    sent = list()
    msg = msg.decode('utf-8')
    blob = TextBlob(msg)
    if(blob.sentiment.polarity > 0):
        # sentiment.append(1)
        sent.append(1)
    elif (blob.sentiment.polarity < 0):
        # sentiment.append(0)
        sent.append(0)
    else:
        # sentiment.append(999)
        sent.append(999)
    sent.append(clas)
    with open('./data/Sahar-results.csv','a') as f:
    # np.savetxt(f,sentiment)
        writer = csv.writer(f)
        writer.writerow(sent)
        # np.savetxt(f, sent)
