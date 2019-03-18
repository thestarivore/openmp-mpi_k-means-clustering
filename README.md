## K-Means Clustering algorithm in OpenMP/MPI

### Requirements:

First install the required libraries:

```
sudo apt install openmpi-bin
sudo apt install libopenmpi-dev
```

**CodeBlocks** can be used to compile the program but with the appropriate changes:

- Create a new copy of the GNU GCC Compiler in "Settings"-->"Compiler" and change the following in "Toolchain Executables":
  - C compiler: mpicc
  - C++ compiler: mpicxx
  - Linker for dynamic libs: mpicxx 
- Add -std=c++03 (to disable c++11) in "Compiler Settings" --> "Other Compiler Settings";
- Add -fopenmp flag (for OpenMP support) in "Project build options" (all tree tabs);

In case of **Manual Compilation** just use the following line:

```
mpicxx main.cpp  -o main -fopenmp
```

### Running Modes:

Run the K-Means Clustering algorithm in four Modes:

- Normal Mode;
- OpenMP Mode;
- MPI Mode;
- MPI + OpenMP Mode;

The first two can be run by running the program normally, the third and forth Modes however require the following command (in the repositories root):

```
mpiexec -n 4 bin/Debug/openmp_cpp_project	//If compiling with CodeBlocks
mpiexec -n 4 ./main							//If compiling from the terminal
```

The program will first ask the number of clusters to use and afterwards will run one Mode at a time by maintaining the initial clusters and dataset common for every execution. 

Once it has finished the execution time will be displayed for each Mode so that we can compare them fairly.

To pass directly the number of centroids to use in the execution, just add the number at the end of the line as an argument:

```
mpiexec -n 4 bin/Debug/openmp_cpp_project N		//If compiling with CodeBlocks
mpiexec -n 4 ./main N							//If compiling from the terminal
```

------

### Ploting

There are three defines in the program that can be decommented to enable three types of plotings:

```
#define PRELOOP_PRINT_AND_PLOT     
#define LOOP_PRINT_AND_PLOT  
#define POSTLOOP_PRINT_AND_PLOT
```

1. **PRELOOP_PRINT_AND_PLOT** enables the plotting of the initial dataset and initial centroids chosen; 

2. **LOOP_PRINT_AND_PLOT** enables the plotting of the dataset, clusters and centroids on every iteration of the execution (can vary based on the initial condition);

3. **POSTLOOP_PRINT_AND_PLOT** enables just the plotting of the final dataset, clusters and centroids;

   

### Amazon AWS - Startcluster

To get meaningful data we need to execute the algorithm on a real cluster, we'll be using Amazon AWS's EC2 service and Starcluster to achieve that. 

Follow the instruction at the following link for a brief guide on how to configure Starcluster on Amazin AWS: http://mpitutorial.com/tutorials/launching-an-amazon-ec2-mpi-cluster/

While for a more detailed document, here is the Starcluster Documentation: https://media.readthedocs.org/pdf/starcluster/latest/starcluster.pdf

The installation process will require the compilation of the ".starcluster/config" file, a copy of the configuration file can be found on the root of the project and it's called **starcluster_config**.

Once the configuration has been completed the following steps must be used to actually execute the algorithm on the cluster and retrieve the results:

1. Create 4x **c3.xlarge** *(4cores and 8GB RAM)* Nodes programmatically on EC2 Amazon AWS  Starcluster to create the cluster (first must be configured). 

   ```
   starcluster terminate -f smallcluster 	//To terminate the last session if any
   starcluster start smallcluster 			//To create the cluster on AWS nodes.
   ```

2. Use the following command to connect via SSH to the master of the clusters:

   ```
   starcluster sshmaster smallcluster 
   ```

3. Use sgeadmin user to control the cluster:

   ```
   su - sgeadmin 
   ```

4. Create the hostfile on sgeadmin root with the following content:

   ```
   master
   node001
   node002
   node003
   ```

5. Clone the git repository on the root folder;

6. Compile and execute:	

   ```
   mpicxx main.cpp  -o main -fopenmp
   mpiexec -np 4 -hostfile /home/sgeadmin/hostfile ./main
   ```

7. Run the script to perform multiple executions:

   ```
   sh multiple_executions.sh 
   ```

8. Save the results;

   

### Testing: Cumulative Results

By running a specific bash script we are able to run multiple executions that can later be used for statistics or any type of data analysis (like the one done with Jupyter Notebook): 

```
sh multiple_executions.sh
```

Plots need to be disabled by commenting the defines described above, otherwise the script will be interrupted by the plots.

### RESULTS: Jupyter Notebook

A Jupyter Notebook documentation is available for data analysis in the Jupyter file **openmp_mpi_k-means.ipynb**(open with Jupyter Notebook) , an  .md file **openmp_mpi_k-means.md**(open with Typora or similar .md readers) and **openmp_mpi_k-means.pdf**.

To download the pdf from the .ipynb you might need to install the following:

```
pip install nbconvert
apt-get install texlive-generic-recommended
sudo apt-get install pandoc

//And use the following command for the convertion
jupyter nbconvert --to pdf MyNotebook.ipynb
```

