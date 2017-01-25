
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

# Split data set for training and testing
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

# print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

# SciKit-Learn pipeline
pipeline = Pipeline([
    # Tokenize strings into integer counts
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    # Convert integer counts into TF-IDF weighted scores
    ('tfidf', TfidfTransformer()),
    # Train Naive Bayes Classifier on TF-IDF vectors
    ('classifier', MultinomialNB()),
])

# Use the pipeline to pre-process, split, train, test, and cross validate.
scores = cross_val_score(pipeline,
                         # Training data
                         msg_train,
                         # Training labels
                         label_train,
                         # Cross-validation slices
                         cv=10,
                         # Choose scoring metric
                         scoring='accuracy',
                         # Use all cores to improve speed
                         n_jobs=-1)

# print(scores)
# print(scores.mean(), scores.std())

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
        Generate a simple plot of the test and traning learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Run classifier and graph accuracy
# plot_learning_curve(pipeline, "Accuracy vs. Training Set Size", msg_train, label_train, cv=10)
# plt.show()

# Examine effect of IDF weighting on accuracy
params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens)
}

grid = GridSearchCV(
    pipeline,
    # Parameters to tune via cross validation
    params,
    # Fit using all available data at the end, on the best found param combination
    refit=True,
    n_jobs=-1,
    # Optimizing this score
    scoring='accuracy',
    # Type of cross validation
    cv=StratifiedKFold(label_train, n_folds=5),
)

# TF-IDF increases accuracy
# nb_detector = grid.fit(msg_train, label_train)
# print(nb_detector.grid_scores_)

# predictions = nb_detector.predict(msg_test)
# print(confusion_matrix(label_test, predictions))
# print(classification_report(label_test, predictions))

# Try using SVM
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),
])

# Pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

# Run the SVM
# svm_detector = grid_svm.fit(msg_train, label_train)
# print(svm_detector.grid_scores_)
# print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
# print(classification_report(label_test,svm_detector.predict(msg_test)))