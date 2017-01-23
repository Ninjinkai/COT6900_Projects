
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

# Inital read of files
# messages = [line.rstrip() for line in open('./data/SMSSpamCollection')]
# print(len(messages))
#
# for message_num, message in enumerate(messages[:10]):
#     print(message_num, message)

messages = pandas.read_csv('./data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
# print(messages)

# Table of message statistics
# print(messages.groupby('label').describe())

# Add message length to the data set
messages['length'] = messages['message'].map(lambda text: len(text))

# Examine message length
# print(messages.head())
# plt.show(messages.length.plot(bins=20, kind='hist'))
# print(messages.length.describe())
# print(list(messages.message[messages.length == 910]))
# print(list(messages.message[messages.length == 2]))

# Compare ham and spam length
# messages.hist(column='length', by='label', bins=50)
# plt.show()

def split_into_tokens(message):
    # message = unicode(message, 'utf8')
    return TextBlob(message).words

# Lemmas are the base form of the word - no caps, inflections, determiners, interjections
def split_into_lemmas(message):
    # message = unicode(message, 'utf8').lower()
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

print(messages.message.head())
print(messages.message.head().apply(split_into_tokens))
print(messages.message.head().apply(split_into_lemmas))