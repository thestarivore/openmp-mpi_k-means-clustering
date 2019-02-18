#include <iostream>
#include <stdlib.h>

#include <stdio.h>
#include <list>
#include <math.h>
#include <time.h>
#include <omp.h>    //OpenMP

using namespace std;


//Typedefs & Structs
typedef struct{
    float x;
    float y;
    int   cn;    //Cluster number
}Point;

typedef enum{
    NORMAL_MODE = 0,
    OPEN_MP_MODE
}ExecMode;

//Defines
//#define PRELOOP_PRINT_AND_PLOT              //Decomment to enable PrintToFile & ClustersPlot before entering the algorithm's loop
//#define LOOP_PRINT_AND_PLOT                 //Decomment to enable PrintToFile & ClustersPlot inside the algorithm's loop
#define POSTLOOP_PRINT_AND_PLOT             //Decomment to enable PrintToFile & ClustersPlot after the algorithm's loop
#define PARALLEL_COMPUTAION                 //Decomment to enable parallel computaion(via OpenMP) of the clusters and centroids recalculation
#define NUMBER_OF_THREADS 8

//Function Prototypes
int     countLines(char *filename);
void    readDataset(char *filename, Point * data);
void    initCentroids(Point * c, int k, Point * data, int ds_rows);
float   distance2Points(Point p1, Point p2);
bool    recalcClusters(Point * c, int k, Point * data, int ds_rows, ExecMode mode);
void    printDataToFile(char *filename, Point * data, int ds_rows, bool newFile);
void    printCentroidsToFile(char *filename, Point * data, int k, bool newFile);
bool    recalcCentroids(Point * c, int k, Point * data, int ds_rows, ExecMode mode);
void    plotClustersFromFile();
float   calcSquaredError(Point * c, int k, Point * data, int ds_rows);
double  kMeansClustering(int k, int n_rows, char newDatasetFile[], char newCentroidsFile[], Point * data, Point * c, ExecMode mode);


int main() {
    char datasetFile[] = "../dataset_display/dataset.csv";
    char newDatasetFile[] = "../dataset_display/newdataset.csv";
    char newCentroidsFile[] = "../dataset_display/newcentroids.csv";
    int k, n_rows;
    //bool centroidsHaveChanged;
    //clock_t start, end;
    //double cpu_time_used;
    double normalExecTime, openMPExecTime;

    //Set number of Threads
    omp_set_num_threads(NUMBER_OF_THREADS);

    //Get number of rows of the dataset (-1 because of the header)
    n_rows = countLines(datasetFile) - 1;
    cout << "Number of rows in the file: " << n_rows << "\n";

    //Allocate the Arrays for the initial dataset
    Point * data = (Point*) malloc(n_rows * sizeof(Point));

    //Read the dataset
    readDataset(datasetFile, data);

    //Print rows in the array
    /*for(int i=0; i<n_rows;i++){
        cout << "X = " << data[i].x << ",Y = " << data[i].y << "\n";
    }*/

    //Pick K
    cout << "Number of centroids (K): ";
    cin >> k;

    //Start time mesurment
 /*   start = clock();

    //Allocate and choose the centroids
    Point * c = (Point*) malloc(k * sizeof(Point));
    initCentroids(c, k, data, n_rows);

    //-------------------------------------------------------------------------------
    //Recalculate Clusters
    recalcClusters(c, k, data, n_rows);

    #ifdef PRELOOP_PRINT_AND_PLOT
        //Print to file the new dataset and the new centroids
        printDataToFile(newDatasetFile, data, n_rows, true);
        printCentroidsToFile(newCentroidsFile, c, k, true);

        //Plot the new dataset & centroids
        plotClustersFromFile();
    #endif // PRELOOP_PRINT_AND_PLOT

    //Centroids recalculation Cicle, stop when the centroids don't change anymore
    do{
        //If data changed
        centroidsHaveChanged = recalcCentroids(c, k, data, n_rows);
        cout << "Changed Centroids..\n ";

        //Recalculate Clusters
        recalcClusters(c, k, data, n_rows);

        #ifdef LOOP_PRINT_AND_PLOT
            //Print to file the new dataset and the new centroids
            printDataToFile(newDatasetFile, data, n_rows, true);
            printCentroidsToFile(newCentroidsFile, c, k, true);

            //Plot the new dataset & centroids
            plotClustersFromFile();
        #endif // LOOP_PRINT_AND_PLOT
    }while(centroidsHaveChanged);

    //End time mesurment - for an accurate time mesurment is strongly recommended to comment
    //                     the PRELOOP_PRINT_AND_PLOT & LOOP_PRINT_AND_PLOT defines
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC ;
    cout << "K-Means Clustering execution time: " << cpu_time_used*1000 << "ms\n";

    //Objective function(J) calculus (Squared Error function) - Calculate an indicator(score) that can be used
    //to choose K(number of centroids). Usually the Knee-rule is used to choose K in the J-K graph.
    float J = calcSquaredError(c, k, data, n_rows);
    cout << "K-Means Clustering squared error: " << J << "\n";

    #ifdef POSTLOOP_PRINT_AND_PLOT
        //Print to file the new dataset and the new centroids
        printDataToFile(newDatasetFile, data, n_rows, true);
        printCentroidsToFile(newCentroidsFile, c, k, true);

        //Plot the new dataset & centroids
        plotClustersFromFile();
    #endif // POSTLOOP_PRINT_AND_PLOT

    //End message
    cout << "Finished! Centroids can't change anymore..\n ";*/

    //Allocate and choose the centroids
    Point * c = (Point*) malloc(k * sizeof(Point));
    initCentroids(c, k, data, n_rows);
    Point * c2 = (Point*) malloc(k * sizeof(Point));
    for(int h=0; h < k; h++){
        (c2+h)->cn = (c+h)->cn;
        (c2+h)->x = (c+h)->x;
        (c2+h)->y = (c+h)->y;
    }

    //omp_set_num_threads(10);

    normalExecTime = kMeansClustering(k, n_rows, newDatasetFile, newCentroidsFile, data, c, NORMAL_MODE);
    openMPExecTime = kMeansClustering(k, n_rows, newDatasetFile, newCentroidsFile, data, c2, OPEN_MP_MODE);

    cout << "K-Means Clustering Normal execution time: " << normalExecTime*1000 << "ms\n";
    cout << "K-Means Clustering OpenMP execution time: " << openMPExecTime*1000 << "ms\n";

    return 0;
}


/**
 * @brief   Run the K-Means Clustering Algorithm
 * @retval  cpu_time_used   Return the execution time of the algorithm on the passed dataset
 */
double kMeansClustering(int k, int n_rows, char newDatasetFile[], char newCentroidsFile[], Point * data, Point * c, ExecMode mode){
    bool centroidsHaveChanged;
    double cpu_time_used;
    double start, end;

    //Start time mesurment
    start = omp_get_wtime();

  /*
    //Allocate and choose the centroids
    Point * c = (Point*) malloc(k * sizeof(Point));
    initCentroids(c, k, data, n_rows);*/

    //-------------------------------------------------------------------------------
    //Recalculate Clusters
    recalcClusters(c, k, data, n_rows, mode);

    #ifdef PRELOOP_PRINT_AND_PLOT
        //Print to file the new dataset and the new centroids
        printDataToFile(newDatasetFile, data, n_rows, true);
        printCentroidsToFile(newCentroidsFile, c, k, true);

        //Plot the new dataset & centroids
        plotClustersFromFile();
    #endif // PRELOOP_PRINT_AND_PLOT

    //Centroids recalculation Cicle, stop when the centroids don't change anymore
    do{
        //If data changed
        centroidsHaveChanged = recalcCentroids(c, k, data, n_rows, mode);
        cout << "Changed Centroids..\n ";

        //Recalculate Clusters
        recalcClusters(c, k, data, n_rows, mode);

        #ifdef LOOP_PRINT_AND_PLOT
            //Print to file the new dataset and the new centroids
            printDataToFile(newDatasetFile, data, n_rows, true);
            printCentroidsToFile(newCentroidsFile, c, k, true);

            //Plot the new dataset & centroids
            plotClustersFromFile();
        #endif // LOOP_PRINT_AND_PLOT
    }while(centroidsHaveChanged);

    //End time mesurment - for an accurate time mesurment is strongly recommended to comment
    //                     the PRELOOP_PRINT_AND_PLOT & LOOP_PRINT_AND_PLOT defines
    end = omp_get_wtime();
    cpu_time_used = ((double) (end - start));// / CLOCKS_PER_SEC ;
    cout << "K-Means Clustering execution time: " << cpu_time_used << "sec\n";

    //Objective function(J) calculus (Squared Error function) - Calculate an indicator(score) that can be used
    //to choose K(number of centroids). Usually the Knee-rule is used to choose K in the J-K graph.
    float J = calcSquaredError(c, k, data, n_rows);
    cout << "K-Means Clustering squared error: " << J << "\n";

    #ifdef POSTLOOP_PRINT_AND_PLOT
        //Print to file the new dataset and the new centroids
        printDataToFile(newDatasetFile, data, n_rows, true);
        printCentroidsToFile(newCentroidsFile, c, k, true);

        //Plot the new dataset & centroids
        plotClustersFromFile();
    #endif // POSTLOOP_PRINT_AND_PLOT

    //End message
    cout << "Finished! Centroids can't change anymore..\n ";
    return cpu_time_used;
}

/**
 * @brief   Run a Python script that plots the clusters from the two files saved
 *          newdataset.cvs & newcentroids.cvs. This function is blocking the execution,
 *          the plot must clused in order to contunue with the normal execution.
 * @retval  None
 */
void plotClustersFromFile(){
    system("python3 ../dataset_display/main.py plot_clusters");
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
    pmin.x = data[0].x;
    pmin.y = data[0].y;
    pmax.x = data[0].x;
    pmax.y = data[0].y;
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
    distThreshold = diagonalDim/15;

    //randomize rand()
    srand(time(NULL));

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
 * @brief   Calculate the distance between two points in 2D space (Euclidean distance)
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
bool recalcClusters(Point * c, int k, Point * data, int ds_rows, ExecMode mode){
    //int i,j;// centroidIndex = 0;
    //float * distances = (float *) malloc(k * sizeof(float));
    //float minDist;
    bool clustersChanged = false;

    //mode = NORMAL_MODE;

    //Iterate all the data points
    //#ifdef PARALLEL_COMPUTAION
    if(mode == OPEN_MP_MODE){
        #pragma omp parallel shared(data, clustersChanged)
        #pragma omp for schedule(static) nowait
        //#endif // PARALLEL_COMPUTAION
        for(int i=0; i<ds_rows; i++){
            int     centroidIndex = 0;
            float   minDist = distance2Points(data[i], c[0]);
            float   distance;

            //Get all the distances from the centroids
            for(int j=0; j<k; j++){
                distance = distance2Points(data[i], c[j]);
                if(minDist > distance){
                    minDist = distance;
                    centroidIndex = j;
                }
            }

            //Set new cluster number of the point
            if(data[i].cn != centroidIndex){
                data[i].cn = centroidIndex;
                clustersChanged = true;
            }
        }
    }else{
        for(int i=0; i<ds_rows; i++){
            int     centroidIndex = 0;
            float   minDist = distance2Points(data[i], c[0]);
            float   distance;

            //Get all the distances from the centroids
            for(int j=0; j<k; j++){
                distance = distance2Points(data[i], c[j]);
                if(minDist > distance){
                    minDist = distance;
                    centroidIndex = j;
                }
            }

            //Set new cluster number of the point
            if(data[i].cn != centroidIndex){
                data[i].cn = centroidIndex;
                clustersChanged = true;
            }
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
bool recalcCentroids(Point * c, int k, Point * data, int ds_rows, ExecMode mode){
    //int i,j;// meanCount = 0;
    //float newCentroidX, newCentroidY;
    bool centroidsChanged = false;
    //float oldX, oldY;

    //Iterate all the centroids
    //#ifdef PARALLEL_COMPUTAION
    if(mode == OPEN_MP_MODE){
        #pragma omp parallel shared(c, centroidsChanged)
        #pragma omp for schedule(static) nowait
        //#endif // PARALLEL_COMPUTAION
        for(int j=0; j<k; j++){
            float   newCentroidX = 0;
            float   newCentroidY = 0;
            float   oldX = c[j].x;
            float   oldY = c[j].y;
            int     meanCount = 0;

            //Iterate all the data points
            for(int i=0; i<ds_rows; i++){
                //The datapoints that belong to this cluster
                if(data[i].cn == j){
                    newCentroidX += data[i].x;
                    newCentroidY += data[i].y;
                    meanCount++;
                }
            }

            //Set new centroid (if no node belogngs to the centroid, then leave it unchanged)
            if(meanCount != 0){ //ZERO Division protection
                c[j].x = newCentroidX / meanCount;
                c[j].y = newCentroidY / meanCount;
            }

            //Control if the centroid has changed
            if(oldX != c[j].x || oldY != c[j].y)
                centroidsChanged = true;
        }
    }else{
        for(int j=0; j<k; j++){
            float   newCentroidX = 0;
            float   newCentroidY = 0;
            float   oldX = c[j].x;
            float   oldY = c[j].y;
            int     meanCount = 0;

            //Iterate all the data points
            for(int i=0; i<ds_rows; i++){
                //The datapoints that belong to this cluster
                if(data[i].cn == j){
                    newCentroidX += data[i].x;
                    newCentroidY += data[i].y;
                    meanCount++;
                }
            }

            //Set new centroid (if no node belogngs to the centroid, then leave it unchanged)
            if(meanCount != 0){ //ZERO Division protection
                c[j].x = newCentroidX / meanCount;
                c[j].y = newCentroidY / meanCount;
            }

            //Control if the centroid has changed
            if(oldX != c[j].x || oldY != c[j].y)
                centroidsChanged = true;
        }
    }

    return centroidsChanged;
}

/**
 * @brief   Objective function(J) calculus (Squared Error function) - Calculate an indicator(score) that can be used
 *          to choose K(number of centroids). Usually the Knee-rule is used to choose K in the J-K graph.
 * @param   c       Point struct centroids array of the x,y positions
 * @param   k       number of centroids
 * @param   data    Point struct data array of the x,y positions
 * @param   ds_rows number of points in the dataset
 * @retval  J function's float value
 */
float calcSquaredError(Point * c, int k, Point * data, int ds_rows){
    int i,j, meanCount = 0;
    float J = 0.0f, sqrDist;

    //Iterate all the centroids
    for(j=0; j<k; j++){
        sqrDist = 0.0f;

        //Iterate all the data points
        for(i=0; i<ds_rows; i++){
            //The datapoints that belong to this cluster
            if(data[i].cn == j){
                sqrDist = pow(distance2Points(data[i], c[j]), 2);
                J += sqrDist;
            }
        }
    }

    return J;
}
