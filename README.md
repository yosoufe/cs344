cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

Install OpenCV3 from [here](https://www.learnopencv.com/install-opencv3-on-ubuntu/).

I had to change all the makefiles to make compatible for opencv3 and my GPU.

# Summary
## Lesson 1
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
