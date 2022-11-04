/* 
  

  Parameters for the model:

    n:      dimenstion of the grid of hospital beds
    k:      number of days of contagion
    tau:    transmission rate for the infected
    delta:  mobility rate of the our_current_gridulation
    nu:     vaccination rate
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <mpi.h>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>
#include <trng/uniform_dist.hpp>

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


void getArguments(int argc, char *argv[], int *n, int *k, float *tau, float *nu, int *rep);
void check_arguments(int n, int k, float tau, float nu);
int assert_minimum_value(char which_value[16], int actual_value, int expected_value);
int assert_maximum_value(char which_value[16], int actual_value, int expected_value);
void pluralize_value_if_needed(int value); 
void printGrid(int **our_current_grid, int nrows, int ncols);
void initialize_grid(int **grid, int nRows, int nCols);
int spread_infection(int **our_current_grid, int **our_next_grid, int nRows, int nCols, int k, float tau, float nu, float delta, int rank);
bool infect(int **our_current_grid, int i, int j, float tau, int nRows, int nCols, float rand);
void vaccinate(int **our_next_grid, int i, int j, float rand, float nu);
void exit_if(int boolean_expression, char function_name[32], int OUR_RANK);

int main(int argc, char **argv) {
  
  //--- default values to be updated by command line argument ------------------- 
  int n = 10;
  int k = 5;
  float tau = .1;
  float nu = .1;
  int rep = 1;
  int intial_infected = 1200;

  //--- the 2d grids of integers -------------------------------------------------
  int **our_current_grid, **our_next_grid;

  //--- loop variables -----------------------------------------------------------
  int row;
  int t;
  int current_row, current_column;
  int new_value;
  int next_lowest_rank, next_highest_rank;

  //--- MPI variables ------------------------------------------------------------
  int rank = 0;
  int num_processes = 1;
  int our_number_rows;
  MPI_Status status;
  
  //--- Time variables -----------------------------------------------------------
  double startTime = 0.0;
  double endTime = 0.0;

  //--- Random variables ---------------------------------------------------------
  double transmission_prob;
  double vax_prob; 
  int rand_x, rand_y;
  double prob;
  double total_prob;

  //--- Infection counts ----------------------------------------------------------
  int my_ninfected = 1; 
  int our_ninfected = 1;
  int my_total_infected = 1;
  int our_total_infected = 1;
  int my_total_vaccinated;
  int our_total_vaccinated;
  //-------------------------------------------------------------------------------

  /* 0.   Initialize the distributed memory environment (MPI)*/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  /* 1.   Parse and check command line arguments */ 
  getArguments(argc, argv, &n, &k, &tau, &nu, &rep);
  check_arguments(n, k, tau, nu);

  /* 2.   Start timing using barrier */
  if (rank == 0) {
    startTime = MPI_Wtime();
  }

  /* 3.   Determine number of rows */
  our_number_rows = n / num_processes + 2;
  if (rank == num_processes - 1) {
    our_number_rows += n % num_processes;
  }

  /* 4.   Allocate enough space in memory for the amount of 
   *      rows, columns, and 2 ghost rows */
  our_current_grid = (int**)malloc((our_number_rows) * n * sizeof(int));
  our_next_grid = (int**)malloc((our_number_rows) * n * sizeof(int));
  for (row = 0; row < our_number_rows + 2; row ++) {
    our_current_grid[row] = (int*)malloc(n * sizeof(int));
    our_next_grid[row] = (int*)malloc(n * sizeof(int));
  }

  /* 5.   Initialize the grid to all 0s */
  for (current_row = 0; current_row < our_number_rows; current_row++) {
    for (current_column = 0; current_column < n; current_column++) {
      our_current_grid[current_row][current_column] = 0;
      our_next_grid[current_row][current_column] = 0;
    }
  }

  /* 6.   Initialize random number generator with a random seed */
  trng::yarn2 r;
  r.seed((long unsigned int)time(NULL));

  /* 7.   Set initial_infected number of infected at random locations */
  int num_infected_rank[num_processes];
  double proportion_at_rank[num_processes];
  int i;
  if (rank == 0) {
    trng::uniform01_dist<> U;
    total_prob = 0;
    
    /* 7.1    Get a random number between 0 and 1 for each process */
    for (i = 0; i < num_processes; i++) {
      prob = U(r);
      total_prob += prob;
      proportion_at_rank[i] = prob;
    }

    /* 7.2    Get a proportion for each rank as the random number
     *  divided by the sum of all the random numbers. This gives
     *  us the proportion of the starting infections happening 
     *  at the given process */


    int total = 0;
    for (i = 0; i < num_processes - 1; i++) {

      prob = proportion_at_rank[i] / total_prob;

      num_infected_rank[i] = (prob * intial_infected);

      total += num_infected_rank[i];

    }
    num_infected_rank[num_processes - 1] = intial_infected - total;
  }
  
  /* 8.   Broadcast the list to each process so each process can easily 
   *  know how many locations to initially infect */
  MPI_Bcast(&num_infected_rank, num_processes, MPI_INT, 0, MPI_COMM_WORLD);

  /* 9.   For each process, get the initial infected. Then set a random x and y,
   *  check if the x and y location had already been infected. If it had not, 
   *  thens set location to infected and move to the next step. If the location
   *  had already been infected then repeat the process without going to the next step*/
  for (i = 0; i < num_infected_rank[rank];) {

    trng::uniform_dist<> rand_y_generator(0, n);
    rand_y = rand_y_generator(r);

    trng::uniform_dist<> rand_x_generator(1, our_number_rows - 1);
    rand_x = rand_x_generator(r);
    
    if (our_current_grid[rand_x][rand_y] == 0) {
      our_current_grid[rand_x][rand_y] = 1;
      i++;
    }
  }

  /* 10.    Run the simulation until the number of infected is 0*/
  next_lowest_rank = rank - 1;
  next_highest_rank = rank + 1;
  t = 0;

  // if (rank == 0) {
  //   printf("%d\t%d\t%d\t0\n", rep, t, intial_infected);
  // } 

  while (our_ninfected > 0) {
    t = t + 1;


    /* 10.1   Set up ghost rows */
    if (rank != 0) {
      /* 10.1.1   If rank is not 0, send the top row to the rank above */
      exit_if(MPI_Send(our_current_grid[1], n, MPI_INT, next_lowest_rank, 0, MPI_COMM_WORLD) != 
                    MPI_SUCCESS, (char*)"MPI_SEND(top row)", rank);
    }

    if (rank != num_processes - 1) {
      /* 10.1.2   If rank is not final process, send bottom row to the rank below */
      exit_if(MPI_Send(our_current_grid[our_number_rows - 2], n, MPI_INT, next_highest_rank, 0, MPI_COMM_WORLD) !=
                    MPI_SUCCESS, (char*)"MPI_SEND(bottom row)", rank);
    }

    if (rank != 0) {
      /* 10.1.3   If rank is not 0, receive the ghost row from the row above */
      exit_if(MPI_Recv(our_current_grid[0], n, MPI_INT, next_lowest_rank, 0, MPI_COMM_WORLD, &status) !=
                    MPI_SUCCESS, (char*)"MPI_RECV(top row)", rank);
    }

    if (rank != num_processes - 1) {
      /* 10.1.4   If rank is not final process, receive ghost row from row below */ 
      exit_if(MPI_Recv(our_current_grid[our_number_rows - 1], n, MPI_INT, next_highest_rank, 0, MPI_COMM_WORLD, &status) !=
                    MPI_SUCCESS, (char*)"MPI_RECV(bottom row)", rank);
    }


    /* 10.2   Spread the infection*/
    trng::uniform01_dist<> u; 
    for (current_row = 1; current_row < our_number_rows; current_row++) {
      for (current_column = 0; current_column < n; current_column++) {
        
        /* 10.2.1   if the value of the location is greater than 0, increment, if 
         *  incremented over the time of infection, set to recovered*/
        transmission_prob = u(r);
        new_value = our_current_grid[current_row][current_column];
        if (new_value > 0) {
          new_value = new_value + 1;

          if (new_value > k) {
              new_value = -1;
          }
        }
        /* 10.2.2   if the value is 0, either infect from surrounding neighbors,
         *  vaccinate, or do nothing. */

          if (new_value == 0) {
              new_value = infect(our_current_grid, current_row, current_column, tau, our_number_rows, n, transmission_prob);
          }
        
          our_next_grid[current_row][current_column] = new_value;

          if (new_value == 0) {
            vax_prob = u(r);
            vaccinate(our_next_grid, current_row, current_column, vax_prob, nu);
          }
      }
    }

    /* 10.3   Set the current grid as the grid for the next timestep and calculate total still
     *  still infected and total overall infected for individual process*/
    my_ninfected = 0;
    my_total_vaccinated = 0;

    int my_recovered = 0;

    for (current_row = 1; current_row < our_number_rows - 1; current_row ++) {
      for (current_column = 1; current_column < n - 1; current_column++) {

        our_current_grid[current_row][current_column] = our_next_grid[current_row][current_column];
        
        if(our_next_grid[current_row][current_column] > 0) {
          my_ninfected++;
        }
        if(our_next_grid[current_row][current_column] == 1) {
          my_total_infected++;
        }
        if(our_next_grid[current_row][current_column] == -1) {
          my_recovered++;
        }
        if(our_next_grid[current_row][current_column] == -2) {
          my_total_vaccinated++;
        }
      }
    }
    
    /* 10.4    Have the processes comunicate and add the total number of infected at 
     *  the time step together */
    MPI_Allreduce(&my_ninfected, &our_ninfected, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int our_recovered = 0;
    MPI_Allreduce(&my_recovered, &our_recovered, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&my_total_vaccinated, &our_total_vaccinated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
      int susceptible = n * n - our_recovered - our_ninfected;
      printf("%d\t%d\t%d\t%d\n", rep,t,our_ninfected, our_total_vaccinated);
    }
  } 

  /* 11.    Deallocate memory from grids */
  for (current_row = our_number_rows + 1; current_row <= 0; current_row--) {
    free(our_current_grid[current_row]);
    free(our_next_grid[current_row]);
  }
  free(our_current_grid);
  free(our_next_grid);

  /* 13.    Have processes communicate to get total infected overall */
  MPI_Allreduce(&my_total_infected, &our_total_infected, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // MPI_Allreduce(&my_total_vaccinated, &our_total_vaccinated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* 14.    End the timing */
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    endTime = MPI_Wtime() - startTime;
    printf("\nTime %lf secs.\n\n", endTime);
    // printf("%lf\t%d\t%d\t",endTime, our_total_infected, t);
  }

  /* 15.    Finalize the MPI environment*/
  MPI_Finalize();

  // if (rank == 0) {
    // printf("total number of days taken was %d\n", t);
    // printf("total number of infected %d\n", our_total_infected);
    // printf("total number of vaccinated %d\n", our_total_vaccinated);
  // }

  return 0;
}


/*
  Calculate the value of the individual at given index in grid

  Variable definitions:
  param int **our_current_grid:    2d array our_current_gridulation grid
  param int n:        dimension of grid
  param int k:        number of days of contagion
  param float tau:    transmission rate for the infection
  param float nu:     vaccination rate
  param int i,j:      i is row index, j is column index
*/
int calculate_new_value(int **our_current_grid, int k, float tau, int nRows, int nCols, float rand, int i, int j) {
    int new_value = our_current_grid[i][j];
    if (new_value > 0) {
        new_value = new_value + 1;

        if (new_value > k) {
            new_value = -1;
        }
    }

    else {
        if (new_value == 0) {
            new_value = infect(our_current_grid, i, j, tau, nRows, nCols, rand);

        }
    } 

    return new_value;
}


/*
  Takes a cell and moves the value to another location if a randomly generated 
  number is less than the probability of moving

  Variable definitions:
  param **our_next_grid, pointer to 2d array
  param i, row index
  param  j, column index
  param rand, random number generated between 0 and 1
  param delta, probability of a cell moving
  param n, dimension of the grid

*/
void newPosition(int** our_next_grid, int i, int j, float delta, int n, float rand){
  if(delta > 0){
    if(rand<delta){
        int inew = floor(rand*n+1);
        int jnew = floor(rand*n+1);
        int tt = our_next_grid[i][j];
        our_next_grid[i][j] = our_next_grid[inew][jnew];
        our_next_grid[inew][jnew] = tt;
    }
  }
}


void exit_if(int boolean_expression, char function_name[32], int OUR_RANK)
{
    if(boolean_expression)
    {
        fprintf(stderr, "Rank %d ", OUR_RANK);

        fprintf(stderr, "ERROR in %s\n", (char*)function_name);
        exit(-1);
    }

    return;
}


/*
  Vaccinate the individual by setting to -2 if rand number is less than vax rate

  Variable definitions:
  param **our_next_grid, pointer to 2d array
  param i, row index
  param  j, column index
  param rand, random number generated between 0 and 1
  param nu, vaccination rate between 0 and 1
*/

void vaccinate(int **our_next_grid, int i, int j, float rand, float nu) {
  if (rand < nu) {
    our_next_grid[i][j] = -2;
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
bool infect(int **our_current_grid, int i, int j, float tau, int nRows, int nCols, float rand) {

    //Tracks whether current cell has been infected
    int t = 0;

    //if not the leftmost wall
    if(i > 0) {
        //if left neighbor is sick
        if(our_current_grid[i-1][j] > 0){
            t = (rand < tau);
        }
    }
    //if i is not the rightmost wall
    if(i < nRows - 1) {
        //if left neighbor is sick
        if(our_current_grid[i+1][j] > 0){
            t = t + (rand < tau);
        }
    }

    if(j > 0) {
        //if left neighbor is sick
        if(our_current_grid[i][j-1] > 0){
            t = t + (rand < tau);
        }
    }

    if(j < nCols - 1) {
      
        if(our_current_grid[i][j+1] > 0){
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
void initialize_grid(int **grid, int nRows, int nCols) {
  int row;
  int column;

  for (row = 0; row < nRows; row ++) {
    for (column = 0; column < nCols; column++) {
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
void getArguments(int argc, char *argv[], int *n, int *k, float *tau, float *nu, int *rep) {
  char *nvalue, *kvalue, *tvalue, *vvalue, *rvalue;
  int c;    // result from getopt calls

  while ((c = getopt(argc, argv, "n:k:t:v:d:r:")) != -1) {
    switch (c) {
      case 'r':
        rvalue = optarg;
        *rep = atoi(rvalue);
        break;
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
void check_arguments(int n, int k, float tau, float nu) {
  int return_value = 0;
  return_value = assert_minimum_value((char*)"n", n, MINIMUM_DIMENSION);

  return_value += assert_minimum_value((char*)"k", k, MINIMUM_K);
  
  return_value += assert_minimum_value((char*)"tau", tau, MINIMUM_TAU); 
  return_value += assert_maximum_value((char*)"tau", tau, MAXIMUM_TAU); 

  return_value += assert_minimum_value((char*)"nu", nu, MINIMUM_NU);  
  return_value += assert_maximum_value((char*)"nu", nu, MAXIMUM_NU); 

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

void printGrid(int **our_current_grid, int nrows, int ncols) {
  int current_row, current_column;
  
  for (current_row = 0; current_row < nrows; current_row++) {

    for (current_column = 0; current_column < ncols; current_column ++) {
      
      printf("  %d  ", our_current_grid[current_row][current_column]);

    }

    printf("\n");
  }
  printf("\n");
}