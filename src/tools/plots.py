import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Convenience function to make boxplots black and white
def decolorize_boxplot(boxplot):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(boxplot[element], color='black')


def plot_confusion_matrix(y_pred, y_true, labels, display_labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )
    disp.plot(cmap=plt.cm.binary, colorbar=False, xticks_rotation=45)
