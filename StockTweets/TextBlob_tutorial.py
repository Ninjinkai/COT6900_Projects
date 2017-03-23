from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

with open('./data/msg-sent-short-no-quotes.csv', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="csv")

    print("Classifier created.")

    print(cl.classify("$T tested Boll Band & 50DMA in long-term uptrend point to short-term bounce last 2 times"))