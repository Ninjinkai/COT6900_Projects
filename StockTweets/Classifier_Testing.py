import pickle

nb_classifier = pickle.load(open('./pickles/nb_classifier.pickle', 'rb'))

# print(nb_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
# print(nb_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
# print(nb_classifier.classify("RSH Will bankrupt February"), "Bearish")
# print(nb_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
# print(nb_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")

dt_classifier = pickle.load(open('./pickles/dt_classifier.pickle', 'rb'))

# print(dt_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), "Bullish")
# print(dt_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), "Bearish")
# print(dt_classifier.classify("RSH Will bankrupt February"), "Bearish")
# print(dt_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), "Bullish")
# print(dt_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), "Bullish")

test = open('./data/msg-sent-short-no-quotes-test.csv', 'r')

# print(nb_classifier.accuracy(test))
# nb_classifier.show_informative_features(10)

print(dt_classifier.accuracy(test))