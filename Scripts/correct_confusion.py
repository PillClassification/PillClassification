import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import dataMngr as dm

labels = ["Atripla","Cymbalta","Epzicom","Lexapro","Prezista","Tivicay","Truvada"]

def makeconfusion(f_true, f_pred, labels):
    y_true = dm.file2Data(f_true)
    y_pred = dm.file2Data(f_pred)
    return confusion_matrix(y_true, y_pred, labels)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y_true))
    plt.xticks(tick_marks, y_true, rotation=45)
    plt.yticks(tick_marks, y_true)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = makeconfusion('/Users/mlaskowski/Documents/PillClassification/SampleData.testingdataoutput','/Users/mlaskowski/Documents/PillClassification/testingOutput', labels)
plt.figure()
plot_confusion_matrix(cm)
plt.show()