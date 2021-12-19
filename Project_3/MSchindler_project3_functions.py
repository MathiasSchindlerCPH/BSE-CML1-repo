import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
  
  
def make_scree_plot(array):
  plt.plot(np.cumsum(array.explained_variance_ratio_ * 100))
  plt.xlabel("Number of components (Dimensions)")
  plt.ylabel("Explained variance (%)")
  plt.xticks(np.arange(0, 105, step=5))
  plt.grid(visible = True)

  fig = plt.gcf()
  fig.set_size_inches(18, 5)

  return plt.show()


def make_feat_importance_plot(coefs, feature_names):
  coef_abs = np.absolute(coefs)
  coef_plot = coef_abs.transpose().flatten()
  coef_plot_df = pd.DataFrame({'feature_name': feature_names, 'coef': coef_plot})
  coef_plot_df = coef_plot_df.sort_values('coef', ascending = False)

  plt.bar(x = coef_plot_df['feature_name'], height = coef_plot_df['coef'])
  plt.xticks(rotation='vertical', fontsize = 12)

  fig = plt.gcf()
  fig.set_size_inches(25, 6)

  return plt.show()


def reweight_proba_multi(pi, q, r):
  pi_rw = pi.copy()
  tot = np.dot(pi, (np.array(q)/r))
  for i in range(len(q)):
    pi_rw[:,i] = (pi[:,i] * q[i] / r) / tot
  return pi_rw


def make_multi_point_pred(array):
  df = pd.DataFrame(array)

  N_rows = len(df.index) 
  point_pred_lst = []

  for i in range(N_rows):
    temp = df.iloc[i,:].idxmax(axis = 0)
    temp += 1
    point_pred_lst.append(temp)

  return pd.Series(point_pred_lst, index = df.index)