import pickle
import pprint

blob_nb_sentiment = pickle.load(open('./pickles/blob_nb.pickle', 'rb'))
blobber_sentiment = pickle.load(open('./pickles/blobber.pickle', 'rb'))
vader_sentiment = pickle.load(open('./pickles/vader.pickle', 'rb'))

for i in range(0, 10):
    pprint.pprint(blob_nb_sentiment[i * 200000])
    pprint.pprint(blobber_sentiment[i * 200000])
    pprint.pprint(vader_sentiment[i * 200000])