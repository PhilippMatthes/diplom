import matplotlib.pyplot as plt


# Convenience function to make boxplots black and white
def decolorize_boxplot(boxplot):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(boxplot[element], color='black')
