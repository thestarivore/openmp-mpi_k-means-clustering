#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <math.h>

using namespace std;

typedef struct{
    float x;
    float y;
    int   cn;    //Cluster number
}Point;

//cout << "Hello world!" << endl;
int     countLines(char *filename);
void    readDataset(char *filename, Point * data);
void    initCentroids(Point * c, int k, Point * data, int ds_rows);
float   distance2Points(Point p1, Point p2);
bool    recalcClusters(Point * c, int k, Point * data, int ds_rows);
void    printDataToFile(char *filename, Point * data, int ds_rows, bool newFile);
void    printCentroidsToFile(char *filename, Point * data, int k, bool newFile);
bool    recalcCentroids(Point * c, int k, Point * data, int ds_rows);


int main() {
    char datasetFile[] = "../dataset_display/dataset.csv";
    char newDatasetFile[] = "../dataset_display/newdataset.csv";
    char newCentroidsFile[] = "../dataset_display/newcentroids.csv";
    int k, n_rows;
    bool centroidsHaveChanged;

    //Get number of rows of the dataset (-1 because of the header)
    n_rows = countLines(datasetFile) - 1;
    cout << "Number of rows in the file: " << n_rows << "\n";

    //Allocate the Arrays for the initial dataset
    Point * data = (Point*) malloc(n_rows * sizeof(Point));

    //Read the dataset
    readDataset(datasetFile, data);

    //Print rows in the array
    for(int i=0; i<n_rows;i++){
        cout << "X = " << data[i].x << ",Y = " << data[i].y << "\n";
    }

    //Pick K
    cout << "Number of centroids (K): ";
    cin >> k;

    //Allocate and choose the centroids
    Point * c = (Point*) malloc(k * sizeof(Point));
    initCentroids(c, k, data, n_rows);

    //-------------------------------------------------------------------------------
    //Recalculate Clusters
    recalcClusters(c, k, data, n_rows);

    //Print to file the new dataset and the new centroids
    printDataToFile(newDatasetFile, data, n_rows, true);
    printCentroidsToFile(newCentroidsFile, c, k, true);

    //Plot the new dataset & centroids
    system("python3 ../dataset_display/main.py plot_clusters");

    //Centroids recalculation Cicle, stop when the centroids don't change anymore
    do{
        //If data changed
        centroidsHaveChanged = recalcCentroids(c, k, data, n_rows);
        cout << "Changed Centroids..\n ";

        //Recalculate Clusters
        recalcClusters(c, k, data, n_rows);

        //Print to file the new dataset and the new centroids
        printDataToFile(newDatasetFile, data, n_rows, true);
        printCentroidsToFile(newCentroidsFile, c, k, true);

        //Plot the new dataset & centroids
        system("python3 ../dataset_display/main.py plot_clusters");
    }while(centroidsHaveChanged);

    //Plot the new dataset & centroids
    system("python3 ../dataset_display/main.py plot_clusters");
    cout << "Finished! Centroids can't change anymore..\n ";

    return 0;
}

/**
 * @brief   Get the number of lines in the file, whose path is passed as argument
 * @param   filename    file to read
 * @retval  lines       rows number in the file
 */
int countLines(char *filename)
{
    // count the number of lines in the file called filename
    FILE *fp = fopen(filename,"r");
    int ch=0;
    int lines=0;

    if (fp == NULL)
        return 0;

    while(!feof(fp))
    {
      ch = fgetc(fp);
      if(ch == '\n')
      {
        lines++;
      }
    }
    fclose(fp);
    return lines;
}

/**
 * @brief   Read the Dataset from the file
 * @param   filename    file to read
 * @param   data        Point struct array of the x,y positions
 * @retval  None
 */
void readDataset(char *filename, Point * data)
{
    FILE *myFile;
    float xval=0.0f, yval=0.0f;
    char c1,c2;

    //Open dataset file
    myFile = fopen(filename, "r");

    //Read the Header first
    fscanf(myFile, "%c,%c ", &c1, &c2);
    if(c1=='X' && c2=='Y'){
        //Read file and store it into the array
        int i=0;
        while (fscanf(myFile, "%f,%f ", &xval, &yval) != EOF){
            data[i].x = xval;
            data[i].y = yval;
            data[i].cn = 0;
            i++;
        }
    }

    //Close File
    fclose(myFile);
}

/**
 * @brief   Initialize the centroids
 * @param   c       Point struct centroids array of the x,y positions
 * @param   k       number of centroids
 * @param   data    Point struct array of the x,y positions
 * @param   ds_rows number of points in the dataset
 * @retval  None
 */
void initCentroids(Point * c, int k, Point * data, int ds_rows){
    Point pmin, pmax, r;
    int i, j, countAttempts = 0;
    float dist, diagonalDim, distThreshold;
    bool tooClose = false;

    cout << "\n Initial centroids: \n";

    //Find space boundaries
    for(i=0; i<ds_rows; i++){
        if(pmin.x > data[i].x)
            pmin.x = data[i].x;
        if(pmin.y > data[i].y)
            pmin.y = data[i].y;
        if(pmax.x < data[i].x)
            pmax.x = data[i].x;
        if(pmax.y < data[i].y)
            pmax.y = data[i].y;
    }

    //Set the minimum distance between the centroids
    diagonalDim = distance2Points(pmin, pmax);
    distThreshold = diagonalDim/8;

    //Set initial centroids
    for(i=0; i<k; i++){
        do{
            r.x = rand() % (int)(pmin.x-pmax.x) + pmin.x;   //random number betwen minx and maxx
            r.y = rand() % (int)(pmin.y-pmax.y) + pmin.y;   //random number betwen miny and maxy
            tooClose = false;

            //Iterate backwords the centroids and valuate if the new centroid is too close to another one
            for(j=i; j>0; j--){
                dist = distance2Points(r, c[j]);
                if(dist < distThreshold){
                    tooClose = true;
                }
            }

            //If K is too high, it might stall, so we have to reduce the distance
            //threshold allowed between centroids
            if(countAttempts++ > 2*k){
                distThreshold /= 1.5f;
                countAttempts = 0;
            }
        }while(tooClose);

        c[i].x=r.x;
        c[i].y=r.y;
        cout << "Centroid: x=" << c[i].x << ", y=" << c[i].y << "\n";
    }
}

/**
 * @brief   Calculate the distance between two points in 2D space
 * @retval  Distance between the two points
 */
float distance2Points(Point p1, Point p2){
    float xd = p2.x - p1.x;
    float yd = p2.y - p1.y;

    return sqrt(xd * xd + yd * yd);
}


/**
 * @brief   Recalculate the clusters (Redistribute the data points to the clusters)
 * @param   c       Point struct centroids array of the x,y positions
 * @param   k       number of centroids
 * @param   data    Point struct data array of the x,y positions
 * @param   ds_rows number of points in the dataset
 * @retval  True if the clusters have change, False otherwize
 */
bool recalcClusters(Point * c, int k, Point * data, int ds_rows){
    int i,j, centroidIndex = 0;
    float * distances = (float *) malloc(k * sizeof(float));
    float minDist;
    bool clustersChanged = false;

    //Iterate all the data points
    for(i=0; i<ds_rows; i++){
        centroidIndex = 0;
        minDist = distance2Points(data[i], c[0]);

        //Get all the distances from the centroids
        for(j=0; j<k; j++){
            distances[j] = distance2Points(data[i], c[j]);
            if(minDist > distances[j]){
                minDist = distances[j];
                centroidIndex = j;
            }
        }

        //Set new cluster number of the point
        if(data[i].cn != centroidIndex){
            data[i].cn = centroidIndex;
            clustersChanged = true;
        }
    }
    return clustersChanged;
}

/**
 * @brief   Print the new dataset to file
 * @param   filename    file to read
 * @param   data        Point struct data array of the x,y positions
 * @param   ds_rows     number of points in the dataset
 * @param   newFile     If true erases the file before writing, otherwize the new data
 *                      gets appended at the end of the file
 * @retval  None
 */
void printDataToFile(char *filename, Point * data, int ds_rows, bool newFile){
    FILE *f;

    if(newFile)
        f = fopen(filename, "w");     //w erase, a append at the end
    else
        f = fopen(filename, "a");     //w erase, a append at the end
    if (f == NULL)
    {
        cout << "Error opening file!\n";
        exit(1);
    }

    /* print the header */
    const char *text = "X,Y,Cluster";
    fprintf(f, "%s\n", text);

    /* print all new data */
    //Iterate all the data points
    for(int i=0; i<ds_rows; i++){
        fprintf(f, "%f,%f,%d\n", data[i].x, data[i].y, data[i].cn);
    }

    fclose(f);
}

/**
 * @brief   Print the new centroids to file
 * @param   filename    file to read
 * @param   c           Point struct centroids array of the x,y positions
 * @param   k           number of points in the dataset(number of clusters)
 * @param   newFile     If true erases the file before writing, otherwize the new data
 *                      gets appended at the end of the file
 * @retval  None
 */
void printCentroidsToFile(char *filename, Point * c, int k, bool newFile){
    FILE *f;

    if(newFile)
        f = fopen(filename, "w");     //w erase, a append at the end
    else
        f = fopen(filename, "a");     //w erase, a append at the end
    if (f == NULL)
    {
        cout << "Error opening file!\n";
        exit(1);
    }

    /* print the header */
    const char *text = "X,Y";
    fprintf(f, "%s\n", text);

    /* print all new centroids */
    //Iterate all the clusters
    for(int i=0; i<k; i++){
        fprintf(f, "%f,%f\n", c[i].x, c[i].y);
    }

    fclose(f);
}

/**
 * @brief   Recalculate the centroids
 * @param   c       Point struct centroids array of the x,y positions
 * @param   k       number of centroids
 * @param   data    Point struct data array of the x,y positions
 * @param   ds_rows number of points in the dataset
 * @retval  True if the centroids have change, False otherwize
 */
bool recalcCentroids(Point * c, int k, Point * data, int ds_rows){
    int i,j, meanCount = 0;
    float newCentroidX, newCentroidY;
    bool centroidsChanged = false;
    float oldX, oldY;

    //Iterate all the centroids
    for(j=0; j<k; j++){
        newCentroidX = 0;
        newCentroidY = 0;
        meanCount = 0;
        oldX = c[j].x;
        oldY = c[j].y;

        //Iterate all the data points
        for(i=0; i<ds_rows; i++){
            //The datapoints that belong to this cluster
            if(data[i].cn == j){
                newCentroidX += data[i].x;
                newCentroidY += data[i].y;
                meanCount++;
            }
        }

        //Set new centroid
        c[j].x = newCentroidX / meanCount;
        c[j].y = newCentroidY / meanCount;

        //Control if the centroid has changed
        if(oldX != c[j].x || oldY != c[j].y)
            centroidsChanged = true;
    }

    return centroidsChanged;
}
