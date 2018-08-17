cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

Install OpenCV3 from [here](https://www.learnopencv.com/install-opencv3-on-ubuntu/).

I had to change all the makefiles to make compatible for opencv3 and my GPU.

# Summary
## Lesson 1: Basic CUDA Programming
```
nvcc -o sample sample.cu
```

```
__global__ void kernelFunction(.....){...} // pointers
// known inside the kernel
threadIdx.x; threadIdx.y; threadIdx.z;
blockDim.x; blockDim.y; blockDim.z; 

blockIdx.x; blockIdx.y; blockIdx.z;
gridDim.x; gridDim.y; gridDim.z;

// Alocate Memory in GPU
cudaMalloc((void **) &d_pointer, N_BYTES);

// copy between GPU and CPU
cudaMemcpy(des_p, source_p, size, type);
// types: cudaMemcpyHostToDevice; cudaMemcpyDeviceToHost; cudaMemcpyDeviceToDevice

// launch the kernel:
kernelFunction<<<dim3, dim3, sharedMem >>>(...);

//free the memory
cudaFree(d_pointer);

```

## Lesson 2: Parallel Communication Pattern && Global/Shared/Local Memory
Parallel Communication Patterns:
* Map - One-to-One
* Gather - Many-to-One
* Scatter - One-to-Many
* Stencil (Data reuse - shared pattern) - Several-to-One
* Transpose (Scatter/Gather, Reorder) like transfering Array of structures (AoS) to structure of arrays (SoA) - One-to-One
* reduce - Many-to-One
* scan/sort - all-to-all

```
__shared__ int array[128];  // example of shared memory
// Shared memory is shared between the threads of a block and 
// it only survives when the block is alive.
__synchthreads();           // make a barrier
```

Writing Efficient GPU Programs:
* Maximise Arithmatic Density. (math ops/memory ops)
* speed: local > shared >> global >> Host Memory
* Parameters are local memory.
* Coalesce Global Memory Accesses.
* Avoid Thread Divergence

```
// defined by this course. Not in CUDA
#include "gputimer.h"
GpuTimer timer;
timer.Start();
timer.Stop();
timer.Elapsed();
```

Atomic Memory Operation: `atomicAdd(); atomicMin(); atomicXOR(); atomicCAS(); ...`

## Lesson 3: Analyse Speed & Efficiency && Reduce, Scan, Histogram
New Terms:
* Step Complexity
* Work Complexity

### Reduce:
Inputs to REDUCE:
1. Set of Elements
2. Operator:
  a. Binary
  b. Associative

### External Shared Memory:
```
// inside the kernel:
extern __shared__ float sharedData[];

// Launch the kernel with the externally defined shared memory
 sharedMem_kernel<<<blocks, threads, size>>>(...);
```

### Exclusive/Inclusive Scan:
Example:
Imput:  1 2 3 4
Operation: Add
Output: 0 1 3 6 10

Inputs to SCAN:
* Input Array
* Binary Associative Operator
* Identity Element

Implementations:
* Hillis/Steele Inclusive Scan: Step: log n, work: n*log n
* Blelloch Scan: Step: 2*log n, work: 2*log n



