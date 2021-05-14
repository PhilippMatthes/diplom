import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# import some data to play with
digits = datasets.load_digits()
X = digits.data
y = digits.target
class_names = digits.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

f, axs = plt.subplots(1, 2, figsize=(15,5))

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=1)

# Plot non-normalized confusion matrix
disp = plot_confusion_matrix(classifier, X_test, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.binary,
                             normalize=None, ax=axs[0])
axs[0].set_title("Konfusionsmatrix (Testdatensatz)")

# Plot non-normalized confusion matrix
disp = plot_confusion_matrix(classifier, X_train, y_train,
                             display_labels=class_names,
                             cmap=plt.cm.binary,
                             normalize=None, ax=axs[1])
axs[1].set_title("Konfusionsmatrix (Trainingsdatensatz)")

plt.savefig('conf-mat-example.pdf', dpi=1200, bbox_inches='tight')
