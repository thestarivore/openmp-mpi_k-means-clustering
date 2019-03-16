#include <iostream>
#include <stdlib.h>

#include <stdio.h>
#include <list>
#include <math.h>
#include <time.h>
#include <omp.h>    //OpenMP Library
#include "mpi.h"    //MPI Library
#include "stddef.h"

using namespace std;


//Typedefs & Structs
typedef struct{
    float x;
    float y;
    int   cn;    //Cluster number
}Point;

typedef enum{
    NORMAL_MODE = 0,    //One process, no parallelization
    OPEN_MP_MODE,       //In this Mode the CPU's cores are used to increase performance by parallelization
    MPI_MODE,           //In this Mode the CPU's cores within a Node are used to increase performance by parallelization
    MPI_OPENMP_MODE,    //In this Mode MPI is used to run a process on each Node(computer), and on each Node OpenMP is used to
                        // take advantage of all the cores within the Node
}ExecMode;

typedef struct{
    float execTime;     //In ms
    float objFunResult;
}KMCResult;

//Defines
//#define PRELOOP_PRINT_AND_PLOT              //Decomment to enable PrintToFile & ClustersPlot before entering the algorithm's loop
//#define LOOP_PRINT_AND_PLOT                 //Decomment to enable PrintToFile & ClustersPlot inside the algorithm's loop
//#define POSTLOOP_PRINT_AND_PLOT             //Decomment to enable PrintToFile & ClustersPlot after the algorithm's loop
#define PARALLEL_COMPUTAION                 //Decomment to enable parallel computaion(via OpenMP) of the clusters and centroids recalculation

//Function Prototypes
int         countLines(char *filename);
void        readDataset(char *filename, Point * data);
void        initCentroids(Point * c, int k, Point * data, int ds_rows);
float       distance2Points(Point p1, Point p2);
bool        recalcClusters(Point * c, int k, Point * data, int ds_rows, ExecMode mode);
void        printDataToFile(char *filename, Point * data, int ds_rows, bool newFile);
void        printCentroidsToFile(char *filename, Point * data, int k, bool newFile);
void        printObjFunctionToFile(char *filename, float objFunResult, bool newFile);
void        printExecTimeToFile(char *filename, float execTime, bool newFile);
void        printResultsToFile(char *filename, int k, ExecMode mode, float execTime, float objFunResult, bool newFile);
bool        recalcCentroids(Point * c, int k, Point * data, int ds_rows, ExecMode mode);
void        plotClustersFromFile();
float       calcSquaredError(Point * c, int k, Point * data, int ds_rows);
KMCResult   kMeansClustering(int k, int n_rows, char newDatasetFile[], char newCentroidsFile[], Point * data, Point * c, ExecMode mode);

int numtasks, rank;
MPI_Datatype mpi_point_type;

int main(int argc, char *argv[]) {
    char datasetFile[]          = "../dataset_display/dataset_100K.csv";
    char initialDatasetFile[]   = "../dataset_display/initialdataset.csv";
    char initialCentroidsFile[] = "../dataset_display/initialcentroids.csv";
    char newDatasetFile[]       = "../dataset_display/newdataset.csv";
    char newCentroidsFile[]     = "../dataset_display/newcentroids.csv";
    char objFunFile[]           = "../dataset_display/objfun.csv";
    char execTimesFile[]        = "../dataset_display/exectimes.csv";
    char resultsFile[]          = "../dataset_display/results.csv";
    int k=-1, n_rows, rc;
    KMCResult normalExecResult, openMPExecResult, mpiExecResult, mpiOpenMPExecResult;
    Point *data, *c, *c2, *c3, *c4;

    //Get the number of centroids as program argument
    if(argv[1] != NULL){
        k = atoi(argv[1]);
        cout << "Number of centroids (K): " << k << endl;
    }

    //Initialize the MPI execution environment
    MPI_Init(0,0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if(rank == 0){
        //Set number of Threads
        omp_set_num_threads(omp_get_max_threads());

        //Get number of rows of the dataset (-1 because of the header)
        n_rows = countLines(datasetFile) - 1;
        cout << "Number of rows in the file: " << n_rows << "\n";

        //Allocate the Arrays for the initial dataset
        data = (Point*) malloc(n_rows * sizeof(Point));

        //Read the dataset
        readDataset(datasetFile, data);

        //Pick K if not already passed at runtime
        if(k == -1){
            cout << "Number of centroids (K): ";
            cin >> k;
        }

        //Allocate and choose the centroids
        c = (Point*) malloc(k * sizeof(Point));
        c2 = (Point*) malloc(k * sizeof(Point));
        c3 = (Point*) malloc(k * sizeof(Point));
        c4 = (Point*) malloc(k * sizeof(Point));
        initCentroids(c, k, data, n_rows);
        for(int h=0; h < k; h++){
            (c2+h)->cn  = (c+h)->cn;
            (c2+h)->x   = (c+h)->x;
            (c2+h)->y   = (c+h)->y;
            (c3+h)->cn  = (c+h)->cn;
            (c3+h)->x   = (c+h)->x;
            (c3+h)->y   = (c+h)->y;
            (c4+h)->cn  = (c+h)->cn;
            (c4+h)->x   = (c+h)->x;
            (c4+h)->y   = (c+h)->y;
        }

        //Print to file the initial dataset(+clusters) and centroids
        printDataToFile(initialDatasetFile, data, n_rows, true);
        printCentroidsToFile(initialCentroidsFile, c, k, true);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //Boardcast the number of rows: n_rows
    rc = MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Task %d: received  n_rows = %d\n", rank, n_rows);

    //Boardcast the number of clusters: K
    rc = MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Task %d: received  k = %d\n", rank, k);

    //Create MPI DataType for Point
    const int    nitems=3;
    int          blocklengths[3] = {1,1,1};
    MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_INT};
    MPI_Aint     offsets[3];

    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);
    offsets[2] = offsetof(Point, cn);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);

    //Boardcast the data
    if(rank != 0)   //Allocate data for the other processes
        data = (Point*) malloc(n_rows * sizeof(Point));
    rc = MPI_Bcast(data, n_rows, mpi_point_type, 0, MPI_COMM_WORLD);

    //Boardcast the clusters
    if(rank != 0){   //Allocate the clusters for the other processes
        c3 = (Point*) malloc(k * sizeof(Point));
        c4 = (Point*) malloc(k * sizeof(Point));
    }
    rc = MPI_Bcast(c3, k, mpi_point_type, 0, MPI_COMM_WORLD);
    rc = MPI_Bcast(c4, k, mpi_point_type, 0, MPI_COMM_WORLD);

    //Execute the K-Means Clustering with three different approaces and record the execution time
    if(rank == 0){
        normalExecResult = kMeansClustering(k, n_rows, newDatasetFile, newCentroidsFile, data, c, NORMAL_MODE);
        openMPExecResult = kMeansClustering(k, n_rows, newDatasetFile, newCentroidsFile, data, c2, OPEN_MP_MODE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    mpiExecResult = kMeansClustering(k, n_rows, newDatasetFile, newCentroidsFile, data, c3, MPI_MODE);
    MPI_Barrier(MPI_COMM_WORLD);
    mpiOpenMPExecResult = kMeansClustering(k, n_rows, newDatasetFile, newCentroidsFile, data, c4, MPI_OPENMP_MODE);


    //Show Execution time for each aproach
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        cout << "K-Means Clustering Normal execution time: "    << normalExecResult.execTime   * 1000 << "ms\n";
        cout << "K-Means Clustering OpenMP execution time: "    << openMPExecResult.execTime   * 1000 << "ms\n";
        cout << "K-Means Clustering MPI execution time: "       << mpiExecResult.execTime      * 1000 << "ms\n";
        cout << "K-Means Clustering MPI+OpenMP execution time: "<< mpiOpenMPExecResult.execTime* 1000 << "ms\n";

        //Print to file the ObjFunction values
        float objFunValue = normalExecResult.objFunResult;
        printObjFunctionToFile(objFunFile, normalExecResult.objFunResult, true);
        printObjFunctionToFile(objFunFile, openMPExecResult.objFunResult, false);
        printObjFunctionToFile(objFunFile, mpiExecResult.objFunResult, false);
        printObjFunctionToFile(objFunFile, mpiOpenMPExecResult.objFunResult, false);

        //Print to file the Execution Time values
        printExecTimeToFile(execTimesFile, normalExecResult.execTime, true);
        printExecTimeToFile(execTimesFile, openMPExecResult.execTime, false);
        printExecTimeToFile(execTimesFile, mpiExecResult.execTime, false);
        printExecTimeToFile(execTimesFile, mpiOpenMPExecResult.execTime, false);

        //Print the total Results (Cumulative with the past results --> for data analisis)
        printResultsToFile(resultsFile, k, NORMAL_MODE,     normalExecResult.execTime, objFunValue, false);
        printResultsToFile(resultsFile, k, OPEN_MP_MODE,    openMPExecResult.execTime, objFunValue, false);
        printResultsToFile(resultsFile, k, MPI_MODE,        mpiExecResult.execTime, objFunValue, false);
        printResultsToFile(resultsFile, k, MPI_OPENMP_MODE, mpiOpenMPExecResult.execTime, objFunValue, false);
    }

    cout << "Process " << rank << " has finished!\n";

    //Terminate the MPI execution environment
    MPI_Finalize();

    return 0;
}


/**
 * @brief   Run the K-Means Clustering Algorithm
 * @retval  result      Returns the KMCResult struct variable containing the execution time and obj function value
 *                      of the algorithm on the passed dataset
 */
KMCResult kMeansClustering(int k, int n_rows, char newDatasetFile[], char newCentroidsFile[], Point * data, Point * c, ExecMode mode){
    bool centroidsHaveChanged;
    double cpu_time_used;
    double start, end;
    KMCResult result;
    MPI_Status stat;
    int rc;
    float J;

    //Start time mesurment
    if(rank == 0)
        start = omp_get_wtime();

    //Step only needed on MPI MODE
    int rowsPerProc, startRow, endRow, numRows;
    Point * sendData;
    if(mode == MPI_MODE || mode == MPI_OPENMP_MODE){
        //-------------------------------------------------------------------------------
        //Recalculate Clusters
        recalcClusters(c, k, data, n_rows, mode);

        //Caculate Start and End row for each task
        rowsPerProc = n_rows / numtasks;
        startRow = rank * rowsPerProc;
        endRow   = startRow + rowsPerProc;
        numRows  = endRow-startRow;
        if(rank == (numtasks-1))
            endRow = n_rows;

        //Allocate and fill the sendData
        sendData = (Point*) malloc(numRows * sizeof(Point));
        for(int i=startRow; i<endRow; i++){
            sendData[i-startRow] = data[i];
        }

        //Gatter data from all proc
        MPI_Gather(sendData, numRows, mpi_point_type, data, numRows, mpi_point_type, 0, MPI_COMM_WORLD);
        //cout << "Process " << rank << " has gathered outside doWhile!\n";
    }

    #ifdef PRELOOP_PRINT_AND_PLOT
        if(rank == 0){
            //Print to file the new dataset and the new centroids
            printDataToFile(newDatasetFile, data, n_rows, true);
            printCentroidsToFile(newCentroidsFile, c, k, true);

            //Plot the new dataset & centroids
            plotClustersFromFile();
        }
    #endif // PRELOOP_PRINT_AND_PLOT

    int clustersPerProc, startCluster, endCluster, numClusters;
    Point * sendClusters;
    //integer array (of length group size) containing the number of elements that are received from each process
    int recvcounts[numtasks];
    //integer array (of length group size). Entry i specifies the displacement(relative to recvbuf) at which to place the incoming data from process i
    int displs[numtasks];

    //Step only needed on MPI MODE
    if(mode == MPI_MODE || mode == MPI_OPENMP_MODE){
        //Caculate Start and End clusters for each task
        clustersPerProc = k / numtasks;
        startCluster = rank * clustersPerProc;
        endCluster   = startCluster + clustersPerProc;
        if(rank == (numtasks-1))        //the last proc gets all the remaining clusters
            endCluster = k;
        numClusters  = endCluster - startCluster;

        //Allocate the sendClusters
        sendClusters = (Point*) malloc(numClusters * sizeof(Point));

        //Fill the two arrays for the MPI_Allgatherv functions, in order to be able to gather data of different size
        for(int l=0; l<numtasks; l++){
            recvcounts[l] = clustersPerProc;
            displs[l] = l * clustersPerProc;
            if(l == (numtasks-1)){
                recvcounts[l] = k - 3 * clustersPerProc;
            }
            //printf("recvcounts[%d] = %d\n", l, recvcounts[l]);
            //printf("displs[%d] = %d\n", l, displs[l]);
        }
    }

    //Centroids recalculation Cicle, stop when the centroids don't change anymore
    do{
        //If data changed
        centroidsHaveChanged = recalcCentroids(c, k, data, n_rows, mode);

        //Step only needed on MPI MODE
        if(mode == MPI_MODE || mode == MPI_OPENMP_MODE){
            //Fill the sendClusters
            for(int i=startCluster; i<endCluster; i++){
                sendClusters[i-startCluster] = c[i];
            }

            //Gatter clusters from all proc
            MPI_Allgatherv(sendClusters, numClusters, mpi_point_type, c, recvcounts, displs, mpi_point_type, MPI_COMM_WORLD);

            //Reduce the centroidsHaveChanged Flag for all proceses
            bool lchc = centroidsHaveChanged;
            MPI_Allreduce(&lchc, &centroidsHaveChanged, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        }

        //if(rank == 0)
        ///    cout << "Changed Centroids..\n ";

        //Recalculate Clusters
        recalcClusters(c, k, data, n_rows, mode);

        //Step only needed on MPI MODE
        if(mode == MPI_MODE || mode == MPI_OPENMP_MODE){
            //Fill the sendData
            for(int i=startRow; i<endRow; i++){
                sendData[i-startRow] = data[i];
            }

            //Gatter data from all proc
            MPI_Allgather(sendData, numRows, mpi_point_type, data, numRows, mpi_point_type, MPI_COMM_WORLD);
            //cout << "Process " << rank << " has gathered inside doWhile!\n";
        }

        #ifdef LOOP_PRINT_AND_PLOT
            if(rank == 0){
                //Print to file the new dataset and the new centroids
                printDataToFile(newDatasetFile, data, n_rows, true);
                printCentroidsToFile(newCentroidsFile, c, k, true);

                //Plot the new dataset & centroids
                plotClustersFromFile();
            }
        #endif // LOOP_PRINT_AND_PLOT
    }while(centroidsHaveChanged);

    if(rank == 0){
        //End time mesurment - for an accurate time mesurment is strongly recommended to comment
        //                     the PRELOOP_PRINT_AND_PLOT & LOOP_PRINT_AND_PLOT defines
        end = omp_get_wtime();
        cpu_time_used = ((double) (end - start));// / CLOCKS_PER_SEC ;
        cout << "K-Means Clustering execution time: " << cpu_time_used << "sec\n";

        //Objective function(J) calculus (Squared Error function) - Calculate an indicator(score) that can be used
        //to choose K(number of centroids). Usually the Knee-rule is used to choose K in the J-K graph.
        J = calcSquaredError(c, k, data, n_rows);
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
    }

    //Broadcast results
    rc = MPI_Bcast(&cpu_time_used,  1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    rc = MPI_Bcast(&J,              1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    result.execTime     = cpu_time_used;
    result.objFunResult = J;
    return result;
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
    bool clustersChanged = false;

    //Iterate all the data points
    if(mode == OPEN_MP_MODE){
        #pragma omp parallel shared(data, clustersChanged)
        #pragma omp for schedule(static)
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
    }else if(mode == MPI_MODE){
        if (numtasks >= 2) {
            int rowsPerProc = ds_rows / numtasks;

            //Claculate Start and End row for each task
            int startRow = rank * rowsPerProc;
            int endRow   = startRow + rowsPerProc;
            if(rank == (numtasks-1))
                endRow = ds_rows;

            //printf("I am process %d out of %d. Processing rows %d-%d \n", rank, numtasks, startRow, endRow);

            //Iterate the rows assigned to each task
            for(int i=startRow; i<endRow; i++){
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
        } else {
            printf("Error, you must specify 4 tasks\n");
        }
    }else if(mode == MPI_OPENMP_MODE){
        if (numtasks >= 2) {
            int rowsPerProc = ds_rows / numtasks;

            //Claculate Start and End row for each task
            int startRow = rank * rowsPerProc;
            int endRow   = startRow + rowsPerProc;
            if(rank == (numtasks-1))
                endRow = ds_rows;

            //printf("I am process %d out of %d. Processing rows %d-%d \n", rank, numtasks, startRow, endRow);

            //Iterate the rows assigned to each task
            #pragma omp parallel shared(data, clustersChanged)
            #pragma omp for schedule(static)
            for(int i=startRow; i<endRow; i++){
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
        } else {
            printf("Error, you must specify 4 tasks\n");
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
 * @brief   Print the new Obj Function result to file
 * @param   filename        file to read
 * @param   objFunResult    result of the obj function
 * @param   newFile         If true erases the file before writing, otherwize the new data
 *                          gets appended at the end of the file
 * @retval  None
 */
void printObjFunctionToFile(char *filename, float objFunResult, bool newFile){
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

    if(newFile){
        /* print the header */
        const char *text = "ObjFun";
        fprintf(f, "%s\n", text);
    }

    /* print */
    fprintf(f, "%f\n", objFunResult);

    fclose(f);
}

/**
 * @brief   Print the new Execution Time result to file
 * @param   filename        file to read
 * @param   execTime        result of the execution time
 * @param   newFile         If true erases the file before writing, otherwize the new data
 *                          gets appended at the end of the file
 * @retval  None
 */
void printExecTimeToFile(char *filename, float execTime, bool newFile){
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

    if(newFile){
        /* print the header */
        const char *text = "Time";
        fprintf(f, "%s\n", text);
    }

    /* print */
    fprintf(f, "%f\n", execTime);

    fclose(f);
}

/**
 * @brief   Print the Result to file (inclusing number of clusters, ExecutionMode, ExecutionTime and ObjFunction value)
 * @param   filename        file to read
 * @param   k               number of centroids
 * @param   mode            Execution Mode
 * @param   execTime        result of the execution time
 * @param   objFunResult    result of the obj function
 * @param   newFile         If true erases the file before writing, otherwize the new data
 *                          gets appended at the end of the file
 * @retval  None
 */
void printResultsToFile(char *filename, int k, ExecMode mode, float execTime, float objFunResult, bool newFile){
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

    if(newFile){
        /* print the header */
        const char *text = "K,Mode,Time,ObjFun";
        fprintf(f, "%s\n", text);
    }

    /* print */
    fprintf(f, "%d,%d,%f,%f\n", k, mode, execTime, objFunResult);

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
    bool centroidsChanged = false;

    //Iterate all the centroids
    if(mode == OPEN_MP_MODE){
        #pragma omp parallel shared(c, centroidsChanged)
        #pragma omp for schedule(static)
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
    }else if(mode == MPI_MODE){
        if (numtasks >= 2) {
            int clustersPerProc = k / numtasks;

            //Calculate Start and End clusters for each task
            int startCluster = rank * clustersPerProc;
            int endCluster   = startCluster + clustersPerProc;
            if(rank == (numtasks-1))        //the last proc gets all the remaining clusters
                endCluster = k;

            //printf("I am process %d out of %d. Processing clusters %d-%d \n", rank, numtasks, startCluster, endCluster);
            for(int j=startCluster; j<endCluster; j++){
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
        } else {
            printf("Error, you must specify 4 tasks\n");
        }
    }else if(mode == MPI_OPENMP_MODE){
        if (numtasks >= 2) {
            int clustersPerProc = k / numtasks;

            //Calculate Start and End clusters for each task
            int startCluster = rank * clustersPerProc;
            int endCluster   = startCluster + clustersPerProc;
            if(rank == (numtasks-1))        //the last proc gets all the remaining clusters
                endCluster = k;

            //printf("I am process %d out of %d. Processing clusters %d-%d \n", rank, numtasks, startCluster, endCluster);
            for(int j=startCluster; j<endCluster; j++){
                float   newCentroidX = 0;
                float   newCentroidY = 0;
                float   oldX = c[j].x;
                float   oldY = c[j].y;
                int     meanCount = 0;

                //Iterate all the data points
                //#pragma omp parallel shared(newCentroidX, newCentroidY, meanCount)
                //#pragma omp for schedule(static)
                #pragma omp parallel for default(shared) reduction(+:newCentroidX,newCentroidY,meanCount)
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
        } else {
            printf("Error, you must specify 4 tasks\n");
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
