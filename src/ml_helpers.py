from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score
import warnings
import matplotlib.pyplot as plt
import numpy as np

def crossValidate(X,y,model,k=5):
  """
  perform k-fold cross-validation using the in-built function
  Shuffle the dataset randomly before splitting to reduce the inherent bias in the data itself
  """
  nf_CV = KFold(n_splits=k, shuffle=True, random_state=101)
  results = []

  # iterate through each train-test split
  for train_idx, test_idx in nf_CV.split(X):

    # Split data into training and testing sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train the model on the current training set
    model.fit(X_train, y_train)

    # Make predictions on the current test set
    y_pred = model.predict(X_test)

    # Record the accuracy for the current fold
    results.append(accuracy_score(y_test, y_pred))
  # Return the mean accuracy across all folds (rounded to 3 decimals)
  return round(np.mean(results),3)

def evaluate(y_true, y_pred):
  """
  Evaluate a classification model using confusion matrix and key performance metrics
  """

  # Suppress unnecessary warnings for clean output
  warnings.filterwarnings("ignore")

  # Compute the confusion matrix for the given observation data and predicted results
  cm = confusion_matrix(y_true, y_pred, labels=['Public', 'Private', 'Active'])

  # Display and visualize the confusion matrix
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Public', 'Private', 'Active'])
  fig, ax = plt.subplots(figsize=(3.5, 3.5))
  disp.plot(ax=ax)

  # Add title and show the confusion matrix plot
  plt.title("Confusion Matrix")
  plt.show()

  # Calculate and print performance metrics of the model
  print('Accuracy:', round(accuracy_score(y_true, y_pred),3))
  print('Precision:', round(precision_score(y_true, y_pred, average='weighted'),3))
  print('Recall:', round(recall_score(y_true, y_pred, average='weighted'),3))
  print('F1:', round(f1_score(y_true, y_pred,average='weighted'),3))
