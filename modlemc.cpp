/* 
  

  Parameters for the model:

    n:      dimenstion of the grid of hospital beds
    k:      number of days of contagion
    tau:    transmission rate for the infected
    delta:  mobility rate of the population
    nu:     vaccination rate
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

/********************************************
 * Need at least this many rows and columns *
 ********************************************/
const int MINIMUM_DIMENSION = 1;
const int MINIMUM_K = 1;
const int MINIMUM_TAU = 0;
const int MAXIMUM_TAU = 1;
const int MINIMUM_DELTA = 0;
const int MAXIMUM_DELTA = 1;
const int MINIMUM_NU = 0;
const int MAXIMUM_NU = 1;


void getArguments(int argc, char *argv[], int *n, int *k, float *tau, float *nu, float *delta);
void check_arguments(int n, int k, float tau, float nu, float delta);
int assert_minimum_value(char which_value[16], int actual_value, int expected_value);
int assert_maximum_value(char which_value[16], int actual_value, int expected_value);
void pluralize_value_if_needed(int value); 
void printGrid(int **our_pop, int n);
void initialize_grid(int **grid, int n);
int spread_infection(int **pop, int **npop, int n, int k, float tau, float nu, float delta);
bool infect(int **pop, int i, int j, float tau, int nRows, int nCols, float rand);
void vaccinate(int **npop, int i, int j, float rand, float nu);
int calculate_new_value(int **pop, int k, float tau, int n, float rand, int i, int j);

int main(int argc, char **argv) {
  
  // default value updated by command line argument
  int n = 5;
  int k = 5;
  float tau = .1;
  float nu = .1;
  float delta = .1;

  //number of time steps
  int t;

  // the 2D grids of integers
  int **pop, **npop;

  // loop variables
  int row;

  // get and check command line arguments
  getArguments(argc, argv, &n, &k, &tau, &nu, &delta);
  check_arguments(n, k, tau, nu, delta);

  // allocate memory for the grid
  pop = (int**)malloc(n * n * sizeof(int));
  npop = (int**)malloc(n * n * sizeof(int));
  for (row = 0; row < n; row ++) {
    pop[row] = (int*)malloc(n * sizeof(int));
    npop[row] = (int*)malloc(n * sizeof(int));
  }

  // initialize the grids so every value is 0
  initialize_grid(pop, n);
  initialize_grid(npop, n);

  // calculate the amount of days it takes for infection to spread
  t = spread_infection(pop, npop, n, k, tau, nu, delta);

  printf("it took %d days for 0 infections\n", t);

  free(pop);
  free(npop);

  return 0;
}


/*
  Spread the infeciton, vaccinate, and move individuals

  Variable definitions:
  param int **pop:    2d array population grid
  param int **npop:   2d array population grid in next timestep
  param int n:        dimension of grid
  param int k:        number of days of contagion
  param float tau:    transmission rate for the infection
  param float nu:     vaccination rate
*/
int spread_infection(int **pop, int **npop, int n, int k, float tau, float nu, float delta) {
  int t, i, j, new_value;

  float rand0, rand1, rand2;
  
  int ninfected = 1;
  trng::mt19937_64 RNengine1;
  trng::uniform_dist<> uni(0, 1);

  pop[1][2] = 1; // set first patient to infected (probably change to random nums later?

  t = 0;

  //printGrid(pop, n);
  while (ninfected > 0) {
    t = t + 1;
    for (i = 0; i < n; i ++) {
      for (j = 0; j < n; j++) {
        
        rand0 = uni(RNengine1); 
        new_value = calculate_new_value(pop, k, tau, n, rand0, i, j);
        npop[i][j] = new_value;

        rand1 = uni(RNengine1);
        vaccinate(npop, i, j, rand1, nu);
      }
    }

    ninfected = 0;

    for (i = 0; i < n; i ++) {
      for (j = 0; j < n; j++) {

        pop[i][j] = npop[i][j];
        
        if(npop[i][j] > 0) {
          ninfected++;
        }
      }
    }
    //printGrid(pop, n);
  } 
  int totalInfected = 0;
   for (i = 0; i < n; i ++) {
      for (j = 0; j < n; j++) {
        if(pop[i][j] ==-1) {
          totalInfected++;
        }
      }
    }
  printf("total infected: %d\n", totalInfected);
  return t;
}


/*
  Calculate the value of the individual at given index in grid

  Variable definitions:
  param int **pop:    2d array population grid
  param int n:        dimension of grid
  param int k:        number of days of contagion
  param float tau:    transmission rate for the infection
  param float nu:     vaccination rate
  param int i,j:      i is row index, j is column index
*/
int calculate_new_value(int **pop, int k, float tau, int n, float rand, int i, int j) {
    int new_value = pop[i][j];
    if (new_value > 0) {
        new_value = new_value + 1;

        if (new_value > k) {
            new_value = -1;
        }
    }

    else {
        if (new_value == 0) {
            new_value = infect(pop, i, j, tau, n, n, rand);

        }
    } 

    return new_value;
}


/*
  Vaccinate the individual by setting to -2 if rand number is less than vax rate

  Variable definitions:
  param **npop, pointer to 2d array
  param i, row index
  param  j, column index
  param rand, random number generated between 0 and 1
  param nu, vaccination rate between 0 and 1
*/

void vaccinate(int **npop, int i, int j, float rand, float nu) {
  if (npop[i][j] == 0 && rand < nu) {
    npop[i][j] = -2;
  }
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
**/
bool infect(int **pop, int i, int j, float tau, int nRows, int nCols, float rand){

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


/*
  Given a 2d array, set every element equal to 0

  Variable definitions:
  param **grid:   pointer to a 2d array
  param n:        dimension of grid
*/
void initialize_grid(int **grid, int n) {
  int row;
  int column;

  for (row = 0; row < n; row ++) {
    for (column = 0; column < n; column++) {
      grid[row][column] = 0;
    }
  }
}

/*
  Get arguments passed by user from the command line

  Variable definitions:
  param *n:     pointer to int variable containing dimension for grid
  param *k:     pointer to int variable containing the days of infection
  param *tau:   pointer to float variable containing the transmission rate
  param *nu:    pointer to float variable containing the vaccination rate
  param *delta: pointer to flaot variable containing the mobility rate

*/
void getArguments(int argc, char *argv[], int *n, int *k, float *tau, float *nu, float *delta) {
  char *nvalue, *kvalue, *tvalue, *vvalue, *dvalue;
  int c;    // result from getopt calls

  while ((c = getopt(argc, argv, "n:k:t:v:d:")) != -1) {
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
      case 'd':
        dvalue = optarg;
        *delta = atof(dvalue);
        break;
      case '?':
      default:
        fprintf(stderr, "Usage %s [-n dimension of grid] [-k number of days in contagion] [-t transmissablitiy rate] [-v vaccination rate] [-d mobility rate]\n", argv[0]);
        
        exit(-1);  
    }
  }
}

/*
  Get arguments passed by user from the command line

  Variable definitions:
  param n:     int variable containing dimension for grid
  param k:     int variable containing the days of infection
  param tau:   float variable containing the transmission rate
  param nu:    float variable containing the vaccination rate
  param delta: flaot variable containing the mobility rate

*/
void check_arguments(int n, int k, float tau, float nu, float delta) {
  int return_value = 0;
  return_value = assert_minimum_value((char*)"n", n, MINIMUM_DIMENSION);

  return_value += assert_minimum_value((char*)"k", k, MINIMUM_K);
  
  return_value += assert_minimum_value((char*)"tau", tau, MINIMUM_TAU); 
  return_value += assert_maximum_value((char*)"tau", tau, MAXIMUM_TAU); 

  return_value += assert_minimum_value((char*)"nu", nu, MINIMUM_NU);  
  return_value += assert_maximum_value((char*)"nu", nu, MAXIMUM_NU); 
  
  return_value += assert_minimum_value((char*)"delta", delta, MINIMUM_DELTA);  
  return_value += assert_maximum_value((char*)"delta", delta, MAXIMUM_DELTA);

  if (return_value != 0) {
    exit(-1);
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

/*******************************************************************************
 * Make sure a value is <= another value, print error and return -1 if it isn't
 ******************************************************************************/
int assert_maximum_value(char which_value[16], int actual_value,
        int expected_value)
{
    int retval;

    if(actual_value > expected_value)
    {
        fprintf(stderr, "ERROR: %d %s", actual_value, which_value);
        pluralize_value_if_needed(actual_value);
        fprintf(stderr, "; need at most %d %s", expected_value, which_value);
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

void printGrid(int **pop, int n) {
  int current_row, current_column;
  
  for (current_row = 0; current_row < n; current_row++) {

    for (current_column = 0; current_column < n; current_column ++) {
      
      printf("  %d  ", pop[current_row][current_column]);

    }

    printf("\n");
  }
  printf("\n");
}


