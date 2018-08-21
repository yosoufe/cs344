# Reduce, Histogram and Scan Patterns

## Comparison of different implementation of HW3:

This is the average of 5 times execution for `memorial_png_large.gold`

### Implementation 1: 2.9697ms
`findMinMax(), cal_hist_atomic(), cal_cdf_serial().`

In `cal_hist_atomic` the atomic operation would be the bottleneck. That means if we have `numBins = 1024`, then 1024 operations is only happening in parallel. Not more than that. We should remove this bottleneck and increase the parallelism. 

### Implementation 2:
Try to optimise histogram calculation. 
* Give each thread, a local histogram.
* Combine the threads with reduction operation within the block.
This means, we need shared memory for the reduction part between all the threads within a block, then do the atomic operation between the blocks.

(For example) Each thread processes `numPixels` (=1024) pixels. We would need `n/numPixels` (= 393216/1024 = 384) threads. That would require `numBins * numThreadsWithinBlock * sizeof(float)` (= 1024*1024 * sizeof(float) = 1048576 * sizeof(float) ).

| Array Size | numPixels Processed in each Thread | Number of threads  | Number of Blocks |Required Shared Memory within each Block ( *sizeof(float) Bytes )
|----------|----------|----------|----------|------|
|393216|1024|384|1|1048576|
|393216|256|1536|2|1048576|
|393216|64|6144|6|1048576|
|393216|32|12288|12|1048576|