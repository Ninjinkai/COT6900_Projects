import pickle

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

vader_sentiment = pickle.load(open('./pickles/vader.pickle', 'rb'))
output_file = open('./data/vader-analysis.txt', 'w')

vader_predictions = []
vader_labels = []

for item in vader_sentiment:
    if item['vader_analysis_pos'] > item['vader_analysis_neg']:
        vader_predictions.append(1)
        vader_labels.append(int(item['stated_sentiment']))
    elif item['vader_analysis_pos'] < item['vader_analysis_neg']:
        vader_predictions.append(0)
        vader_labels.append(int(item['stated_sentiment']))

fpr, tpr, thresholds = metrics.roc_curve(vader_labels, vader_predictions, pos_label=1)
roc_auc = auc(fpr, tpr)
accu = accuracy_score(vader_labels, vader_predictions)
pre = precision_score(vader_labels, vader_predictions)
rec = recall_score(vader_labels, vader_predictions)
fone = f1_score(vader_labels, vader_predictions)
rocScore = roc_auc_score(vader_labels, vader_predictions)

output_file.write(
    'Vader Sentiment Analyzer:'
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