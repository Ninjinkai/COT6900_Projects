import sys
import twitter
import pprint
from twitter import *
import matplotlib
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import _pickle as cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve

input_csv = open('./data/msg-sent.csv', 'r')
output_csv = open('./data/msg-sent-new.csv', 'w')
output_csv_01 = open('./data/msg-sent-01.csv', 'w')

tweets = []

output_csv.write('body,sentiment\n')
output_csv_01.write('body,sentiment\n')

for line in input_csv:
    line = line[1:].replace(' \'', '\'').replace('   ', ' ').replace('  ', ' ')
    output_csv.write(line)
    output_csv_01.write(line.replace(',Bullish', ',1').replace(',Bearish', ',0'))
    tweets.append(line.replace('\'', '').strip().split(','))

print(tweets[0:4])

input_csv.close()
output_csv.close()
output_csv_01.close()