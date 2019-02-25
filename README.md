**K-Means Clustering algorithm in OpenMP/MPI**

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

