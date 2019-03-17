import csv
import random
import sys

import numpy
import os  # We need this module
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.datasets.samples_generator import make_blobs
sns.set()  # for plot styling

#Samples
N_SAMPLES = 100

# Get path of the current dir, then use it to create paths:
CURRENT_DIR = os.path.dirname(__file__)
file_path = os.path.join(CURRENT_DIR, 'dataset.csv')
new_dataset_path = os.path.join(CURRENT_DIR, 'newdataset.csv')
new_centroids_path = os.path.join(CURRENT_DIR, 'newcentroids.csv')

def main():
    plotType = ""

    # print command line arguments
    for arg in sys.argv[1:]:
        plotType = arg
        print(arg)

    if plotType == "":
        plotDataSet()
    elif plotType == "create_new_dataset":
        createDataSet()
    elif plotType == "plot_dataset":
        plotDataSet()
    elif plotType == "plot_clusters":
        plotNewDataSet()


def createDataSet():
    X, y_true = make_blobs(n_samples=N_SAMPLES, centers=4,
                           cluster_std=0.60, random_state=3320)
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.show()

    with open(file_path, 'w') as csvfile:
        fieldnames = ['X', 'Y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for x in range(N_SAMPLES):
            writer.writerow({'X': X[x, 0], 'Y': X[x, 1]})

def plotDataSet():
    x = numpy.zeros(N_SAMPLES)
    y = numpy.zeros(N_SAMPLES)

    #Read the dataset from the CVS file
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            x[i] = row['X']
            y[i] = row['Y']
            #print(x[i], y[i])
            i=i+1

    #Plot the read dataset
    plt.scatter(x[:], y[:], s=50)
    plt.show()


def plotNewDataSet():
    x = numpy.zeros(N_SAMPLES)
    y = numpy.zeros(N_SAMPLES)
    c = numpy.zeros(N_SAMPLES)
    cx = list()
    cy = list()

    # Read the new centroids from the CVS file
    with open(new_centroids_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cx.append(float(row['X']))
            cy.append(float(row['Y']))

    # Read the new dataset from the CVS file
    with open(new_dataset_path) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            x[i] = row['X']
            y[i] = row['Y']
            c[i] = row['Cluster']
            # print(x[i], y[i])
            i = i + 1

    minK = c.min()
    maxK = c.max()
    k = (int)(maxK - minK + 1)

    # plot the points for each cluster with a different color
    for i in range(k):
        x2 = list()
        y2 = list()

        for j in range(N_SAMPLES):
            if c[j] == i:
                x2.append(x[j])
                y2.append(y[j])

        # Plot the read dataset
        color1 = random_color()
        color2 = random_color()
        plt.scatter(x2[:], y2[:], c=color1, s=50)
        plt.scatter(cx[i], cy[i], c=color1, marker="X", edgecolor=color2, s=100)
    plt.show()


def random_color():
    return numpy.random.rand(3,)


if __name__== "__main__":
    main()
