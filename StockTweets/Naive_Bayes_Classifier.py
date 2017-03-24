from textblob.classifiers import NaiveBayesClassifier
import pickle

train = open('./data/msg-sent-short-no-quotes.csv', 'r')
test = open('./data/msg-sent-short-no-quotes-test.csv', 'r')

# Naive Bayes
nb_classifier = NaiveBayesClassifier(train, format="csv")

print("Naive Bayes classifier created.")

# print(nb_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
# print(nb_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
# print(nb_classifier.classify("RSH Will bankrupt February"), "Bearish")
# print(nb_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
# print(nb_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")

print(nb_classifier.accuracy(test))
nb_classifier.show_informative_features(10)

train.close()
test.close()

nb_pickle_file = open('./pickles/nb_classifier.pickle', 'wb')
pickle.dump(nb_classifier, nb_pickle_file)
nb_pickle_file.close()