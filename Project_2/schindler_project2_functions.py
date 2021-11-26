import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Jack's confusion matrix plotter
def plot_confusion_matrix(cm, class_labels):
    """Pretty prints a confusion matrix as a figure

    Args:
        cm:  A confusion matrix for example
        [[245, 5 ], 
         [ 34, 245]]
         
        class_labels: The list of class labels to be plotted on x-y axis

    Rerturns:
        Just plots the confusion matrix.
    """
    
    df_cm = pd.DataFrame(cm, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
    sns.set(font_scale=1)
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("Real label")
    plt.show()


## Correction factor
def reweight(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w


def make_roc_plot(fpr, tpr, roc_auc):
  formatter = "{0:.3f}"
  plt.plot(fpr, tpr, label='ROC curve (area =' + formatter.format(roc_auc) + ')')

  plt.plot([0, 1], [0, 1], 'k--') 
  plt.title('Receiver Operating Characteristic')
  plt.xlabel('False Positive Rate or (1 - Specifity)')
  plt.ylabel('True Positive Rate or (Sensitivity)')
  plt.legend(loc="lower right")

  fig = plt.gcf()
  fig.set_size_inches(14, 8)
  plt.show()