/* 
  HODGE-C -- A C implementation of Martin Gerhard & Heike Schuster's hodge-podge machine.

  This is a version drastically cut down by John Burkardt.
  The only other file it needs is "hodge.map", which is used to pick the colors.

  Parameters for the model:

    n:    dimenstion of the grid of hospital beds
    k:    number of days of contagion
    tau:  transmission rate for the infected
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <iostream>
#include <fstream>

#include <trng/mt19937_64.hpp>
#include <trng/mt19937.hpp>
#include <trng/lcg64_shift.hpp>
#include <trng/normal_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/exponential_dist.hpp>
#include <trng/uniform01_dist.hpp>

#include "openacc.h"
#include "fillRand.cu"
#include <curand.h>
#include <time.h>
#include <omp.h> 


/********************************************
 * Need at least this many rows and columns *
 ********************************************/
const int MINIMUM_DIMENSION = 1;
const int MINIMUM_K = 0;
const int MINIMUM_TAU = 0;

void getArguments(int argc, char *argv[], int *n, int *k, float *tau, float *nu);
int assert_minimum_value(char which_value[16], int actual_value, int expected_value);
void pluralize_value_if_needed(int value); 
void printGrid(int **our_pop, int n);
void initialize_grid(int **grid, int n);
void spread_infection(int **pop, int **npop, int n, int k, float tau, float nu, float delta);
bool infect(int **pop, int i, int j, float tau, int nRows, int nCols, float rand);
float* fillRandNumArray(int n, int timesteps);
void vaccinate(int **npop, int i, int j, float rand, float nu);
void newPosition(int** npop, int i, int j, float delta, int n, float rand);
int calcInfected(int** pop, int n);


curandGenerator_t setup_prng(void *stream, unsigned long long seed);
void gen_rand_nums(curandGenerator_t gen, float *d_buffer, int num, void *stream);
void rand_cleanup( curandGenerator_t gen );

using namespace std;
int main(int argc, char **argv) {
  //Change to one big array with number of cells times # of iterations
  // trng::mt19937_64 RNengine1;
  // trng::uniform01_dist<> uni;

  //for testing
  // float rand = uni(RNengine1);
  // float rand2 = uni(RNengine1);

  // printf("rand = %f, rand2 = %f\n", rand, rand2);
  
  // default value updated by command line argument
  int n = 10;
  int k = 5;
  float tau = .1;
  float nu = .3;
  float delta = .1;

  // for checking if n, k, tau are sensible
  int return_value;

  // the 2D grids of integers
  int **pop, **npop;

  // loop variables
  int row;

  // set random engine
  
  // get command line arguments
  getArguments(argc, argv, &n, &k, &tau, &nu);

  // make sure dimension is not <= 0
  return_value = assert_minimum_value("n", n, MINIMUM_DIMENSION);
  // make sure number of days of contagion is not < 1
  return_value += assert_minimum_value("k", k, MINIMUM_K);
  // make sure tau is >= 0
  return_value += assert_minimum_value("tau", tau, MINIMUM_TAU);

  if (return_value != 0) {
    exit(-1);
  }
  
  pop = (int**)malloc(n * n * sizeof(int));
  npop = (int**)malloc(n * n * sizeof(int));
  for (row = 0; row < n; row ++) {
    pop[row] = (int*)malloc(n * sizeof(int));
    npop[row] = (int*)malloc(n * sizeof(int));
  }
  double st = omp_get_wtime();
  initialize_grid(pop, n);
  initialize_grid(npop, n);
  spread_infection(pop, npop, n, k, tau, nu, delta);
  double runtime = omp_get_wtime() - st;
  printf("total runtime: %f\n", runtime);
  free(pop);
  free(npop);
}

void spread_infection(int **pop, int **npop, int n, int k, float tau, float nu, float delta) {
  int t, i, j, new_value;
  float rando, rand3;
  srand(time(0));
  //infect 10 random people at the start
  if(n > 9){
    int i=0;
    while(i<10){
      int rand1 = rand() % n;
      int rand2 = rand() % n;
      pop[rand1][rand2] = 1;
      i++;
    }
  }

  int startInfected = 0;
  //need to count because not random index might be referenced multiple times when setting initial infected
  #pragma acc kernels
    #pragma acc loop independent
    for (i = 0; i < n; i ++) {
      #pragma acc loop independent
      for (j = 0; j < n; j++) {
        if(pop[i][j] == 1){
          startInfected++;
        }
      }
    }
   // set first patient to infected (probably change to random nums later?

  void *stream = acc_get_cuda_stream(acc_async_sync);
    int length = (n)*(n);
    curandGenerator_t cuda_gen;
    float *restrict arrayRN = (float*)malloc(length*sizeof(float));

    // use CUDA library functions to initialize a generator
    unsigned long long seed = time(NULL);
    cuda_gen = setup_prng(stream, seed);
  int ninfected = startInfected;

  t = 0;

  //printGrid(pop, n);
  while (ninfected >0) {
    t = t + 1;
    stream = acc_get_cuda_stream(acc_async_sync);
    #pragma acc host_data use_device(arrayRN)
    {
        gen_rand_nums(cuda_gen, arrayRN, length, stream);
    }
    //Start Parallelization
    //Lets make the grid count in the millions so that we can actually parallelize
    #pragma acc kernels
    #pragma acc loop independent
    for (i = 0; i < n; i ++) {
      #pragma acc loop independent private(new_value, i, j, rando) reduction(+:ninfected)
      for (j = 0; j < n; j++) {
        new_value = pop[i][j];
        if (new_value > 0) {
          new_value = new_value + 1;

          if (new_value > k) {
            new_value = -1;
            ninfected--;
          }

        }

        else {
          if (new_value == 0) {
            rando = arrayRN[i*n+j];//+(t-1)*n*n
            //printf("%f\n", rando);
            new_value = infect(pop, i, j, tau, n, n, rando);
            if(new_value == 1){
              ninfected++;
            }
          }
        }
        rand3 = arrayRN[j*n+i];
        //rand4 = arrayRN[(j+1)*(i+1)-1];
        npop[i][j] = new_value;
        vaccinate(npop, i, j, rand3, nu);
        //newPosition(npop, i, j, delta, n, rand4);
      }


    }
    
    #pragma acc kernels
    #pragma acc loop independent
    for (i = 0; i < n; i ++) {
      #pragma acc loop independent
      for (j = 0; j < n; j++) {

        pop[i][j] = npop[i][j];

      }
    }
    //printGrid(pop, n);
  } 
  rand_cleanup(cuda_gen);
  
  //Calculate total Infected
  int totalInfected = calcInfected(pop, n);
  
    printf("total infected: %d\n", totalInfected);
}
//only works at the end not the start
int calcInfected(int** pop, int n){
  int i, j;
  int totalInfected=0;
  #pragma acc kernels
    #pragma acc loop independent
    for (i = 0; i < n; i ++) {
      #pragma acc loop independent
      for (j = 0; j < n; j++) {
        if(pop[i][j] == -1){
          totalInfected++;
        }
      }
    }
    return totalInfected;
}
/**
Looks at all of a current cell's neighbors and gets a random num for each
infected neighbor and generates a random num from 0 to 1, then compares that
num to the infection rate to determine if it gets infected.
Variable Definitions:
i = current x pos
j = current y pos
nRows = numRows
nCols = numColumns
tau = Transmission Rate

Not Parallelizable
**/
bool infect(int **pop, int i, int j, float tau, int nRows, int nCols, float rand){
    //printf("rand = %f", rand);

    //Tracks whether current cell has been infected
    int t = 0;

    //if not the leftmost wall
    if(i > 0) {
        //if left neighbor is sick
        if(pop[i-1][j] > 0){
            t = (rand < tau);
        }
    }
    //if i is not the rightmost wall
    if(i < nRows - 1) {
        //if left neighbor is sick
        if(pop[i+1][j] > 0){
            t = t + (rand < tau);
        }
    }

    if(j > 0) {
        //if left neighbor is sick
        if(pop[i][j-1] > 0){
            t = t + (rand < tau);
        }
    }

    if(j < nCols - 1) {
      
        if(pop[i][j+1] > 0){
            t = t + (rand < tau);
        }
    }

    bool p = 0;
    
    if(t > 0){
        p = 1;
    }
    
    return p;

}
void newPosition(int** npop, int i, int j, float delta, int n, float rand){
  if(delta > 0){
    if(rand<delta){
        int inew = floor(rand*n+1);
        int jnew = floor(rand*n+1);
        int tt = npop[i][j];
        npop[i][j] = npop[inew][jnew];
        npop[inew][jnew] = tt;
    }
  }
}


void vaccinate(int **npop, int i, int j, float rand, float nu) {
  if (npop[i][j] == 0 && rand < nu) {
    npop[i][j] = -2;
  }
}

//Could be parallelizable, but if a small grid, unneccesary.
void initialize_grid(int **grid, int n) {
  int row;
  int column;
  #pragma acc kernels
    #pragma acc loop independent
  for (row = 0; row < n; row ++) {
    #pragma acc loop independent
    for (column = 0; column < n; column++) {
      grid[row][column] = 0;
    }
  }
}

//Not Parallelizable
void getArguments(int argc, char *argv[], int *n, int *k, float *tau, float *nu) {
  char *nvalue, *kvalue, *tvalue, *vvalue;
  int c;    // result from getopt calls

  while ((c = getopt(argc, argv, "n:k:t:v:")) != -1) {
    switch (c) {
      case 'n':
        nvalue = optarg;
        *n = atoi(nvalue);
        break;
      case 'k':
        kvalue = optarg;
        *k = atoi(kvalue);
        break;
      case 't':
        tvalue = optarg;
        *tau = atof(tvalue);
        break;
      case 'v':
        vvalue = optarg;
        *nu = atof(vvalue);
        break;
      case '?':
      default:
        fprintf(stderr, "Usage %s [-n dimension of grid] [-k number of days in contagion] [-t transmissablitiy rate] [-v vaccination rate] \n", argv[0]);
        
        exit(-1);  
    }
  }
}

/*******************************************************************************
 * Make sure a value is >= another value, print error and return -1 if it isn't
 ******************************************************************************/
int assert_minimum_value(char which_value[16], int actual_value,
        int expected_value)
{
    int retval;

    if(actual_value < expected_value)
    {
        fprintf(stderr, "ERROR: %d %s", actual_value, which_value);
        pluralize_value_if_needed(actual_value);
        fprintf(stderr, "; need at least %d %s", expected_value, which_value);
        pluralize_value_if_needed(expected_value);
        fprintf(stderr, "\n");
        retval = -1;
    }
    else
        retval = 0;

    return retval;
}

/*****************************************************
 * Add an "s" to the end of a value's name if needed *
 *****************************************************/
void pluralize_value_if_needed(int value)
{
    if(value != 1)
        fprintf(stderr, "s");

    return;
}
//Probably don't need to parallelize
void printGrid(int **pop, int n) {
  int current_row, current_column;
  
  for (current_row = 0; current_row < n; current_row++) {

    for (current_column = 0; current_column < n; current_column ++) {
      
      printf("%d", pop[current_row][current_column]);

    }

    printf("\n");
  }
  printf("\n");
}


