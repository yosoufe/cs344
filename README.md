cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit. 
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

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
