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

t = Twitter(
    auth=OAuth(
        '386563588-uGsNSg2T7lgC5awaeFy6mvoQict3yYY29OBFFqDA',
        'qHudO6e3NwXKRtuR2yfCPvqTkqaIbeOlEUhUocJCBZe6G',
        '2PKlDplAmZs5GrE4Nh6J0s1f0',
        'BWMrI7zjCz0ez9H3ITikBfvxhiemmajoAxbyArxOBSTzAORn7w'
               ),
    retry=True
)

output_file = open("search_results.txt", mode='w')
output_file.write(pprint.pformat(t.search.tweets(q="#AAPL")))
output_file.close()

output_file = open("search_results_text.txt", mode='w')
for status in t.search.tweets(q="#AAPL")['statuses']:
    output_file.write(pprint.pformat(status['text']))
output_file.close()