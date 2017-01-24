
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

# print(messages.message.head())
# print(messages.message.head().apply(split_into_tokens))
# print(messages.message.head().apply(split_into_lemmas))

# Convert data to vectors with bag of words model
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
# print(len(bow_transformer.vocabulary_))

# Examining one message
message4 = messages['message'][3]
# print(message4)
bow4 = bow_transformer.transform([message4])
# print(bow4)
# print(bow4.shape)
# Which words appear twice?
# print(bow_transformer.get_feature_names()[6736])
# print(bow_transformer.get_feature_names()[8013])

# Transform into a matrix
messages_bow = bow_transformer.transform(messages['message'])
# print("Sparse matrix shape:", messages_bow.shape)
# print("Sparse matrix cells:", (messages_bow.shape[0] * messages_bow.shape[1]))
# print("Number of non-zeros:", messages_bow.nnz)
# print("Number of zeros:", ((messages_bow.shape[0] * messages_bow.shape[1]) - messages_bow.nnz))
# print("Sparsity: %.2f%%" % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

# Transform into TFIDF
tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Examine one message
tfidf4 = tfidf_transformer.transform(bow4)
# print(tfidf4)

# Examine IDF of 'u' and 'university'
# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
# print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

# Transform bag of words matrix into TFIDF matrix
messages_tfidf = tfidf_transformer.transform(messages_bow)
# print(messages_tfidf.shape)

# Create Naive Bayes classifier
spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])

# Classify the previously examined message
# print("Predicted:", spam_detector.predict(tfidf4)[0])
# print("Expected:", messages.label[3])

# Classify the first 10 messages
# first_10_predicted = []
# first_10_expected = []
# for message_num in range(0, 10):
#     current_message = messages['message'][message_num]
#     message_bow = bow_transformer.transform([current_message])
#     message_tfidf = tfidf_transformer.transform(message_bow)
#     first_10_predicted.append(str(spam_detector.predict(messages_tfidf)[0]))
#     first_10_expected.append(str(messages.label[message_num]))
#
# hit = 0
# miss = 0
# for message_num in range(0, 10):
#     print("Message " + str(message_num) + ": predicted " + first_10_predicted[message_num] +
#     ", expected " + first_10_expected[message_num])
#     if first_10_predicted[message_num] == first_10_expected[message_num]:
#         hit += 1
#     else:
#         miss += 1
#
# print(str(hit) + " hits, " + str(miss) + " misses.")

# Examine the classification
all_predictions = spam_detector.predict(messages_tfidf)
#
# print("Accuracy: ", accuracy_score(messages['label'], all_predictions))
# print("Confusion matrix\n", confusion_matrix(messages['label'], all_predictions))
# print("row = expected, column = predicted")
#
# plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
# plt.title('Confusion matrix')
# plt.colorbar()
# plt.ylabel('Expected label')
# plt.xlabel('Predicted label')
# plt.show()
#
# print(classification_report(messages['label'], all_predictions))