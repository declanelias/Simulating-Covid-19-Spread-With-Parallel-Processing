# Simulating COVID-19 Spread with Parallel Programming

This code provides 3 versions of an infectious outbreak spread amongst a population. Given a dimension size, infection length, transmissability rate, and vaccination rate, the program will spread the infection and run until there are no more infected. The code is based off of Matlab code written by Dianne L. O'Leary in 2004 for her computer science class at the University of Maryland.

## Sequential version
The first version is a sequential version. It can be compiled with

    gcc -o modelmc modelmc.cpp -I /usr/local/include/trng -l trng

and can be run with

    ./modelmc -n <dimension> -k <infection length> -t <transmissability rate> -v <vaccination rate>

## MPI version
The MPI version can be compiled with

    mpic++ modlemc_mpi.cpp -o modlemc_mpi -I /usr/local/include/trng -l trng

and can be run with 
    mpirun -np <number of processes> ./modlemc_mpi -n <dimension> -k <infection length> -t <transmissability rate> -v <vaccination rate> -t <trial number (for testing)>

## GPU version
    The OpenACC GPU version can be compiled with:
    
    pgc++ -fast -ta=tesla:cc75,managed -Minfo=accel -lcurand -Mcuda -o GPUmodle modlemc_openacc.cpp
    
    and can be run with:
    
    ./GPUmodle -n <dimension> -k <infection length> -t <transmissability rate> -v <vaccination rate>

## Analysis

[Analysis](https://docs.google.com/document/d/1syEkue6cQk2FNfN7PeGRPmreaU_iHkjwF8Soh_ehRv0/edit?usp=sharing)
