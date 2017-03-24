from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier, NLTKClassifier
import pickle

train = open('./data/msg-sent-short-no-quotes.csv', 'r')
test = open('./data/msg-sent-short-no-quotes-test.csv', 'r')

# Decision Tree
dt_classifier = DecisionTreeClassifier(train, format="csv")

print("Decision Tree classifier created.")

# print(dt_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
# print(dt_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
# print(dt_classifier.classify("RSH Will bankrupt February"), "Bearish")
# print(dt_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
# print(dt_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")

print(dt_classifier.accuracy(test))

train.close()
test.close()

dt_pickle_file = open('./pickles/dt_classifier.pickle', 'wb')
pickle.dump(dt_classifier, dt_pickle_file)