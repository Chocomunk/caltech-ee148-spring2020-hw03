import os
import numpy as np

import matplotlib.pyplot as plt


''' Problem 7

Plot training and test error as a function of the number of training examples
on a log-log scale.

The data written into this script comes from the training output for the
relevant models.
'''
if __name__=="__main__":
    # Rows: 1/2, 1/4, 1/8, 1/16 model errors
    sizes = np.array([25503, 12750, 6376, 3189])

    # Cols: Train set accuracy, Test set accuracy
    accuracies = np.array([
        [25324. / 25503, 9926. / 10000],
        [12629. / 12750, 9886. / 10000],
        [6298. / 6376, 9845. / 10000],
        [3134. / 3189, 9802. / 10000]
    ])

    # Cols: Train set loss, Test set loss
    losses = np.array([
        [0.0232, 0.0235],
        [0.0308, 0.0344],
        [0.0375, 0.0454],
        [0.0590, 0.0624]
    ])
    error = 1 - accuracies

    plt.figure()

    plt.loglog(sizes, error[:,0], label="Training Set")
    plt.loglog(sizes, error[:,1], label="Test Set")

    plt.legend()
    plt.title("Error (frac incorrect) vs. Training Set Size")
    plt.xlabel("Training Set Size")

    plt.savefig("problem7.png")
    plt.show()