import pickle

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

blob_nb_sentiment = pickle.load(open('./pickles/blob_nb.pickle', 'rb'))
output_file = open('./data/blobber-NB-analysis.txt', 'w')

blob_nb_predictions = []
blob_nb_labels = []

for item in blob_nb_sentiment:

    if item['blob_NB_classification'] == 'pos':
        blob_nb_predictions.append(1)
    else:
        blob_nb_predictions.append(0)
    blob_nb_labels.append(int(item['stated_sentiment']))

fpr, tpr, thresholds = metrics.roc_curve(blob_nb_labels, blob_nb_predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
accu = accuracy_score(blob_nb_labels, blob_nb_predictions)
pre = precision_score(blob_nb_labels, blob_nb_predictions)
rec = recall_score(blob_nb_labels, blob_nb_predictions)
fone = f1_score(blob_nb_labels, blob_nb_predictions)
rocScore = roc_auc_score(blob_nb_labels, blob_nb_predictions)

output_file.write(
    'Blobber Naive Bayes Sentiment Analyzer:'
    '\nAccuracy: ' + str(accu) +
    '\nPrecision: ' + str(pre) +
    '\nRecall: ' + str(rec) +
    '\nF-1: ' + str(fone) +
    '\nROC Score: ' + str(rocScore)
)
output_file.close()

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()