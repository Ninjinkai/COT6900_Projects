import pickle

dt_classifier = pickle.load(open('./pickles/dt_classifier.pickle', 'rb'))

# print(dt_classifier.classify("AAPL I looking jan ER press release remind s Yeah werent impressive time different"), " should be Bullish")
# print(dt_classifier.classify("pmguy No reason AMZN buy RSH assets Real estate leases"), " should be Bearish")
# print(dt_classifier.classify("RSH Will bankrupt February"), " should be Bearish")
# print(dt_classifier.classify("BBRY Blackberry cant fail support mechanisms place years This bust gets traction Very Soon"), " should be Bullish")
# print(dt_classifier.classify("FB Analysis helps identify market weakness move take position FB dipping thanks holiday trading next week"), " should be Bullish")

test = open('./data/msg-sent-short-no-quotes-test.csv', 'r')

print(dt_classifier.accuracy(test))

test.close()