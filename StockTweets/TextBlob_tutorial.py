from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier, NLTKClassifier

train = open('./data/msg-sent-short-no-quotes.csv', 'r')
test = open('./data/msg-sent-short-no-quotes-test.csv', 'r')

# Naive Bayes
nb_classifier = NaiveBayesClassifier(train, format="csv")

print("Naive Bayes classifier created.")

print(nb_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
print(nb_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
print(nb_classifier.classify("RSH Will bankrupt February"), "Bearish")
print(nb_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
print(nb_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")

print(nb_classifier.accuracy(test))
nb_classifier.show_informative_features(10)

# Decision Tree
dt_classifier = DecisionTreeClassifier(train, format="csv")

print("Decision Tree classifier created.")

print(dt_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
print(dt_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
print(dt_classifier.classify("RSH Will bankrupt February"), "Bearish")
print(dt_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
print(dt_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")

print(dt_classifier.accuracy(test, format="csv"))

# nltk_classifier = NLTKClassifier(train, format="csv")
#
# print("NLTK classifier created.")
#
# print(nltk_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
# print(nltk_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
# print(nltk_classifier.classify("RSH Will bankrupt February"), "Bearish")
# print(nltk_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
# print(nltk_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")
#
# print(nltk_classifier.accuracy(test, format="csv"))

train.close()
test.close()