import csv
import numpy
import os  # We need this module
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.datasets.samples_generator import make_blobs
sns.set()  # for plot styling

#Samples
N_SAMPLES = 300

# Get path of the current dir, then use it to create paths:
CURRENT_DIR = os.path.dirname(__file__)
file_path = os.path.join(CURRENT_DIR, 'dataset.csv')


def main():
    #createDataSet()
    plotDataSet()


def createDataSet():
    X, y_true = make_blobs(n_samples=N_SAMPLES, centers=4,
                           cluster_std=0.60, random_state=0)
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
    xyz = numpy.array(numpy.random.random((N_SAMPLES, 3)))

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


if __name__== "__main__":
    main()