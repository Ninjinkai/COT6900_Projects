import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

output_file = open('./data/vader-analysis-01.txt', 'w')

DataSet = pd.read_csv("./data/vader-new-01.csv",header=None, names=['sentWordNet','label'])
predictions = [1 if x=='1' else 0 for x in DataSet['sentWordNet'].tolist()]
label = [1 if x=='1' else 0 for x in DataSet['label'].tolist()]

fpr, tpr, thresholds = metrics.roc_curve(label, predictions, pos_label=1)
roc_auc = auc(fpr,tpr)
accu = accuracy_score(label, predictions)
pre = precision_score(label, predictions)
rec = recall_score(label, predictions)
fone = f1_score(label, predictions)
rocScore = roc_auc_score(label, predictions)

output_file.write(
    'Vader Sentiment Analyzer:'
    '\nNumber of instances: ' + str(len(label)) +
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