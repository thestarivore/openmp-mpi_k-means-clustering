## K-Means Clustering algorithm in OpenMP/MPI

First install the required libraries:

```
sudo apt install openmpi-bin
sudo apt install libopenmpi-dev
```

Run the K-Means Clustering algorithm in three Modes:

- Normal Mode;
- OpenMP Mode;
- MPI Mode;

The first two can be run by running the program normally, the third Mode however needs the following command (in the repositories root):

```
mpiexec -n 4 bin/Debug/openmp_cpp_project
```

The program will first ask the number of clusters to use and afterwards will run one Mode at a time by maintaining the initial clusters and dataset common for every execution. 

Once it has finished the execution time will be displayed for each Mode so that we can compare them fairly.

To pass directly the number of centroids to use in the execution, just add the number at the end of the line as an argument:

```
mpiexec -n 4 bin/Debug/openmp_cpp_project N
```

------

### Ploting

There are three defines in the program that can be decommented to enable three types of plotings:

```
#define PRELOOP_PRINT_AND_PLOT     
#define LOOP_PRINT_AND_PLOT  
#define POSTLOOP_PRINT_AND_PLOT
```

- **PRELOOP_PRINT_AND_PLOT** enables the plotting of the initial dataset and initial centroids chosen; 
- **LOOP_PRINT_AND_PLOT** enables the plotting of the dataset, clusters and centroids on every iteration of the execution (can vary based on the initial condition);
-  **POSTLOOP_PRINT_AND_PLOT** enables just the plotting of the final dataset, clusters and centroids;

### Testing: Cumulative Results

By running a specific bash script we are able to run multiple executions that can later be used for statistics or any type of data analysis (like the one done with Jupyter Notebook): 

```
sh multiple_executions.sh
```

Plots need to be disabled by commenting the defines described above, otherwise the script will be interrupted by the plots.

### RESULTS: Jupyter Notebook

A Jupyter Notebook documentation is available for data analysis in the Jupyter file **openmp_mpi_k-means.ipynb**(open with Jupyter Notebook) and  .md file **openmp_mpi_k-means.md**(open with Typora or similar .md readers).