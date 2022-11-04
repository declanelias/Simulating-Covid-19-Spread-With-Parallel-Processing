/*
 * Generates an array of random numbers
 * Reference: sortGPU.cu https://www.olcf.ornl.gov/tutorials/openacc-interoperability-ii/
 */

#include <stdio.h>
#include <curand.h>

// Fill d_buffer with num random numbers
// If you only need to generate on set of numbers and fill the array once,
// use this function. If you want to fill the array over and over again,
// use the other functions given below.
//
extern "C" void fill_rand(float *d_buffer, int num, void *stream, unsigned long long seed)
{
  curandGenerator_t gen;
  int status = CURAND_STATUS_SUCCESS;

  // Create generator
  status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  // Set CUDA stream
  status |= curandSetStream(gen, (cudaStream_t)stream);

  // Set seed
  status |= curandSetPseudoRandomGeneratorSeed(gen, seed);

  // Generate num random numbers
  // From documentation:
  //The curandGenerateUniform() function is used to generate uniformly
  // distributed floating point values between 0.0 and 1.0, 
  // where 0.0 is excluded and 1.0 is included.
  status |= curandGenerateUniform(gen, d_buffer, num);

  // Cleanup generator
  status |= curandDestroyGenerator(gen);

  if (status != CURAND_STATUS_SUCCESS) {
      printf ("curand failure!\n");
      exit (EXIT_FAILURE);
  }
}


//
// Set up a CUDA random number generator and return it.
//
extern "C" curandGenerator_t setup_prng(void *stream, unsigned long long seed) {
  curandGenerator_t gen;
  int status = CURAND_STATUS_SUCCESS;

  // Create generator
  status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  // Set CUDA stream
  status |= curandSetStream(gen, (cudaStream_t)stream);

  // Set seed
  status |= curandSetPseudoRandomGeneratorSeed(gen, seed);

  if (status != CURAND_STATUS_SUCCESS) {
      printf ("curand failure!\n");
      exit (EXIT_FAILURE);
  }

  return gen;
}

//
// Place a set of random numbers between 0.0 and 1.0 in an array d_buffer.
// This is designed so that with one generator (gen), this function can
// be called multiple times as needed to get a new set of random numbers
// for an iteration of a simulation, for example.
//
extern "C" void gen_rand_nums(curandGenerator_t gen, float *d_buffer, int num, void *stream) {
  int status = CURAND_STATUS_SUCCESS;

  // Generate num random numbers
  // From documentation:
  //The curandGenerateUniform() function is used to generate uniformly
  // distributed floating point values between 0.0 and 1.0, 
  // where 0.0 is excluded and 1.0 is included.
  status |= curandGenerateUniform(gen, d_buffer, num);

  if (status != CURAND_STATUS_SUCCESS) {
      printf ("curand failure!\n");
      exit (EXIT_FAILURE);
  }
}

//
// Remove the CUDA random number generator when finished with it.
//
extern "C" void rand_cleanup( curandGenerator_t gen ) {
  int status = CURAND_STATUS_SUCCESS;

  // Cleanup generator
  status |= curandDestroyGenerator(gen);

  if (status != CURAND_STATUS_SUCCESS) {
      printf ("curand failure!\n");
      exit (EXIT_FAILURE);
  }
}