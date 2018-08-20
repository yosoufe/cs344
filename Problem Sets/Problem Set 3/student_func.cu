/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"
#include <float.h>

// sample of REDUCE pattern in CUDA for min and max operation
__global__
void findMinMax(const float* const d_logLuminance,
                float *min, float *max,
                int logLuminance_size){

  extern __shared__ float s_var[];
  //extern __shared__ float s_min[];
//  if (gridDim.x != 384 || blockDim.x != 1024)
//    printf(" gridDim.x %d, blockDim.x %d, block.x %d, thread.x %d \n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
  float* s_min = (float *) s_var;
  float* s_max = (float *) &s_min[blockDim.x];

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tid = threadIdx.x;
  if (idx >= logLuminance_size) return;

  // copy data to the shared memory from the global memory
  s_min[tid] = d_logLuminance[idx];
  s_max[tid] = d_logLuminance[idx];
  __syncthreads();


  for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
    if (tid < stride){
      s_min[tid] = fminf(s_min[tid], s_min[tid+stride]);
      s_max[tid] = fmaxf(s_max[tid], s_max[tid+stride]);
    }
  }
  if(tid == 0){
    min[blockIdx.x] = s_min[0];
    max[blockIdx.x] = s_max[0];
  }
  __syncthreads();
//  // how many elements in min max are valid:
//  // arraysize/1024 = gridDim.x
  if(tid>=gridDim.x)return;
  s_min[tid] = min[tid];
  s_max[tid] = max[tid];
  __syncthreads();
  for(int s = gridDim.x ; s > 1; ){
    if (s%2 == 0 ) s/=2;
    else s = s/2+1;
    if (tid<s){
      s_min[tid] = fminf(s_min[tid], s_min[tid+s]);
      s_max[tid] = fmaxf(s_max[tid], s_max[tid+s]);
    }
  }
  if (tid == 0 ){
    min[0] = fminf(s_min[0],s_min[1]);
    max[0] = fmaxf(s_max[0],s_max[1]);
  }
}

__global__
void calculate_cdf(const float* const d_logLuminance,
                   float const min, float const max, float const range,
                   unsigned int* const d_cdf,  unsigned int* const d_hist,
                   size_t const numBins,
                   unsigned int const logLuminance_size)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
//  const int tid = threadIdx.x;
  if (idx >= logLuminance_size) return;

  if (idx < numBins) {
    d_hist[idx] = 0;
    d_cdf[idx] = 0;
  }

  // Calculate Histogram
  // bin = (lum[i] - lumMin) / lumRange * numBins
  unsigned int bin = (d_logLuminance[idx] - min) / range * numBins;
  if (bin < numBins) atomicAdd(&d_hist[bin], 1);

  __syncthreads();

  // calculate the cfd
  if (idx >= numBins) return;
  for (int i = 0; i < idx; i++){
    d_cdf[idx] += d_hist[i];
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf, // cumulative distribution funciton
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  /******* 1) find the minimum and maximum *********/
  const int maxThreadPerBlock = 1024;
  const int arraySize = numRows * numCols;
  std::cout << "array size: " << arraySize << std::endl;

  float originalData[arraySize];
  cudaMemcpy(originalData,d_logLuminance,arraySize*sizeof(float),cudaMemcpyDeviceToHost);
  float *d_Min;
  float *d_Max;
  checkCudaErrors(cudaMalloc((void **) &d_Min, sizeof(float) * arraySize));
  checkCudaErrors(cudaMalloc((void **) &d_Max, sizeof(float) * arraySize));
  dim3 blockWidth(maxThreadPerBlock);
  dim3 nOfBlocks(arraySize/maxThreadPerBlock);
  findMinMax<<<nOfBlocks,blockWidth,2*maxThreadPerBlock*sizeof(float)>>>(d_logLuminance,
                                                                       d_Min,d_Max,
                                                                       arraySize);
  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  float h_min, h_max;
  cudaMemcpy(&h_min,d_Min,sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_max,d_Max,sizeof(float),cudaMemcpyDeviceToHost);
  checkCudaErrors(cudaGetLastError());

  cudaFree(d_Min);
  cudaFree(d_Max);

  min_logLum = h_min;
  max_logLum = h_max;

  /* To check the correctness of the GPU implementation:
  float* h_min = (float*) malloc(int(arraySize/maxThreadPerBlock) * sizeof(float));
  float* h_max = (float*) malloc(int(arraySize/maxThreadPerBlock) * sizeof(float));
  cudaMemcpy(h_min,d_Min,sizeof(float)*int(arraySize/maxThreadPerBlock),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_max,d_Max,sizeof(float)*int(arraySize/maxThreadPerBlock),cudaMemcpyDeviceToHost);
  float cpu_min = FLT_MAX;
  float gpu_min = FLT_MAX;
  float cpu_max = FLT_MAX * (-1);
  float gpu_max = FLT_MAX * (-1);
  int max_idx, min_idx;
  for (int i = 0; i < arraySize; i++){
    if (originalData[i] < cpu_min) cpu_min = originalData[i];
    if (originalData[i] > cpu_max) cpu_max = originalData[i];
  }
  for (int i = 0; i < arraySize/maxThreadPerBlock; i++){
    if (h_min[i] < gpu_min) {
      gpu_min = h_min[i];
      min_idx = i;
    }
    if (h_max[i] > gpu_max) {
      gpu_max = h_max[i];
      max_idx = i;
    }
  }

  std::cout << "min: CPU: " << cpu_min << " GPU: " <<
               h_min[0] << " GPU2: " << gpu_min << " at: " << min_idx << std::endl;
  std::cout << "max: CPU: " << cpu_max << " GPU: " <<
               h_max[0] << " GPU2: " << gpu_max << " at: " << max_idx << std::endl;
  delete h_min;
  delete h_max;
  */

  // Calculate the range
  float lumRange = h_max - h_min;
  std::cout << "min: " << h_min << ", max: " << h_max << std::endl;

  // memory allocation for histogram:
  unsigned int* d_hist;
  cudaMalloc((void **)&d_hist, sizeof (unsigned int) * numBins);

  // generate the cumulative distribution funciton d_cdf
  calculate_cdf<<<nOfBlocks,blockWidth>>>(d_logLuminance,
                                          min_logLum,max_logLum,lumRange,
                                          d_cdf,d_hist,
                                          numBins,
                                          arraySize);

  checkCudaErrors(cudaGetLastError());
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
