#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10
do
    for j in 1 2 3 4 5 6 7 8 9 10
    do
	   #mpiexec -n 4 bin/Debug/openmp_cpp_project $i
            mpiexec -np 4 -hostfile /home/sgeadmin/hostfile ./main $i
    done
done
