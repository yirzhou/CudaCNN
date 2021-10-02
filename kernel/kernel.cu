#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <math.h>
#include <stdio.h>

#define INPUT_DIM 100
#define NUM_INPUT_PIXEL 10000
#define CONV_KERNEL_DIM 5
#define NUM_CONV_KERNEL_PIXEL 25
#define CONV_OUT_DIM 20
#define NUM_PIXEL_PER_FEATURE 400
#define NUM_OUT_LAYER_ELEM 4000
#define NUM_NEURON 10
#define BLOCK_DIM_X 256
#define RELU(val) val > 0.0 ? val : 0.0

/* Performs 5x5 convolution and returns the rectified result.
- image: 100 x 100 input image grid
- kernel: 5 x 5 kernel
- start_row: starting y-coordinate of the input
- start_col: starting y-coordinate of the input
*/
__device__ double conv_and_relu(const double *image, const double *kernels,
                                size_t neuron_number, size_t start_row,
                                size_t start_col) {
  double result = 0.0;
  for (size_t i = 0; i < CONV_KERNEL_DIM; ++i) {
    size_t cur_row = start_row + i;
    for (size_t j = 0; j < CONV_KERNEL_DIM; ++j) {
      result += image[cur_row * INPUT_DIM + start_col + j] *
                kernels[neuron_number * NUM_CONV_KERNEL_PIXEL +
                        i * CONV_KERNEL_DIM + j];
    }
  }
  return RELU(result);
}

// Performs dot product between a 1x4000 and 4000x1 matrix.
// - input: 4000x1 intermediate results (or 1x4000 depending on your
// perspective) vector
// - weight: 4000x1 weight matrix
// - out: 1x10 output vector
extern "C" __global__ void output_layer(const double *input,
                                        const double *weights, double *out) {

  // Compute the product between each two elements first.
  // The products will be summed up via reduction later.
  // For example, thread 0 of block 0 will compute on the first weight and the
  // intermediate results (0, 256, 512, 768, 1024, ... if blockDim.x==256).
  __shared__ double partial_sums[BLOCK_DIM_X];
  size_t tid = threadIdx.x;

  double sum = 0.0;
  for (size_t i = tid; i < NUM_OUT_LAYER_ELEM; i += BLOCK_DIM_X) {
    sum += weights[blockIdx.x * NUM_OUT_LAYER_ELEM + i] * input[i];
  }
  partial_sums[tid] = sum;
  // Synchronize all threads within the block before proceeding.
  __syncthreads();

  // Parallel sum reduction:
  // We start with the largest stride to minimize bank conflicts.
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    // Change the indexing to be sequential threads.
    // Each thread does work unless the index goes off the block.
    if (tid < s) {
      partial_sums[tid] += partial_sums[tid + s];
    }
    __syncthreads();
  }

  // Only one thread can write to the final result.
  if (tid == 0) {
    out[blockIdx.x] = partial_sums[0];
  }
}

/* Performs ths simple 3-layer CNN.
- image: 100 x 100 input image grid
- conv_kernels: 10 x 20 x 20 kernels for convolution
- weights: 10 x 1 X 4000 weight matrices
- hidden_layer: 1x4000 flat matrix to store intermediate results
- out: 1 x 10 output matrix
*/
extern "C" __global__ void cnn(const double *image, const double *conv_kernels,
                               double *hidden_layer) {
  // Compute and store the conv'ed & ReLU'ed results into a 1*4000 flat array.
  // Each blockDim = 32 x 32
  if (threadIdx.x < CONV_OUT_DIM && threadIdx.y < CONV_OUT_DIM &&
      blockIdx.x < NUM_NEURON) {
    size_t start_row = threadIdx.y * CONV_KERNEL_DIM;
    size_t start_col = threadIdx.x * CONV_KERNEL_DIM;
    size_t out_idx = blockIdx.x * NUM_PIXEL_PER_FEATURE +
                     threadIdx.y * CONV_OUT_DIM + threadIdx.x;

    hidden_layer[out_idx] =
        conv_and_relu(image, conv_kernels, blockIdx.x, start_row, start_col);
  }
}

// Parallel sum reduction algorithm using shared memory.
// This was used to verify my logic.
extern "C" __global__ void sum_reduction(const double *input, double *output) {
  __shared__ double partial_sums[BLOCK_DIM_X];
  // Synchronize all threads within the block before proceeding.
  __syncthreads();
  double sum = 0.0;
  for (size_t i = threadIdx.x; i < NUM_OUT_LAYER_ELEM; i += BLOCK_DIM_X) {
    sum += input[i];
  }
  partial_sums[threadIdx.x] = sum;
  __syncthreads();

  // Increase the stride of the access until we exceed the CTA dimensions
  for (int s = 1; s < blockDim.x; s *= 2) {
    // Change the indexing to be sequential threads
    int index = 2 * s * threadIdx.x;

    // Each thread does work unless the index goes off the block
    if (index < blockDim.x) {
      partial_sums[index] += partial_sums[index + s];
    }
    __syncthreads();
  }

  // Only one thread can write to the final result.
  if (threadIdx.x == 0) {
    output[blockIdx.x] = partial_sums[0];
  }
}
