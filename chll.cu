#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "murmur3.cu"

// Lengths of an input line, a 36 character UUID
#define INPUT_LINE_WIDTH 36

// Lengths of an process line, a 36 character
// UUID followed by 4 padding bytes for alignment
#define PROCESS_LINE_WIDTH 40

// Default enough space for 100K entries
#define INITIAL_SIZE 100000

// Multiple the buffer size by 4 each time
#define BUF_MULT 4

// Width of our hash function output
#define HASH_WIDTH 16

// Number of threads per block
#define THREADS_PER_BLOCK 192

// Define the number of HLL buckets, and prefix bits to use
#define HLL_BUCKETS 1024
#define HLL_PREFIX_BITS 10  // log2(HLL_BUCKETS)

// How wide is each bucket
#define HLL_BUCKET_WIDTH 6
#define HLL_MAX_SCAN 64     // 2**HLL_BUCKET_WIDTH

#define TWO_32 4294967296 // 2**32

static double alpha() {
    return 0.7213/(1 + 1.079/HLL_BUCKETS);
}


static double raw_estimate(unsigned int *inp) {
    double multi = alpha() * HLL_BUCKETS * HLL_BUCKETS;
    double inv_sum = 0;
    for (int i=0; i<HLL_BUCKETS;i++) {
        inv_sum += 1 / pow(2.0, (int)inp[i]);
    }
    return (1 / inv_sum) * multi;
}


static double range_corrected(double raw, unsigned int *inp) {
    if (raw < (5/2)*HLL_BUCKETS) {
        int numzero = 0;
        for (int i=0; i < HLL_BUCKETS; i++) {
            if (inp[i] == 0) numzero++;
        }
        if (numzero == 0)
            return raw;
        else
            return HLL_BUCKETS * log(HLL_BUCKETS / numzero);

    } else if (raw > (1/30)*TWO_32) {
        return -1*TWO_32*log(1 - (raw / TWO_32));
    } else {
        return raw;
    }
}


__host__ int read_input(char **inp, int *inp_len) {
    // Get the initial buffer
    int buf_size = INITIAL_SIZE * PROCESS_LINE_WIDTH;
    char *buf = (char*)malloc(buf_size);

    int offset = 0;
    int in;
    while (1) {
        in = read(STDIN_FILENO, buf+offset, INPUT_LINE_WIDTH+1);
        if (in == 0) break;
        else if (in < 0) {
            perror("Failed to read input!\n");
            free(buf);
            return 1;
        } else if (in == INPUT_LINE_WIDTH + 1) {
            offset += INPUT_LINE_WIDTH;
            *(buf+offset) = 0;
            *(buf+offset+1) = 0;
            *(buf+offset+2) = 0;
            *(buf+offset+3) = 0;
            offset += 4;
        } else {
            printf("Input is not %d byte aligned!\n", INPUT_LINE_WIDTH);
            free(buf);
            return 1;
        }

        // Check if we need to resize
        if (offset + PROCESS_LINE_WIDTH >= buf_size) {
            char *new_buf = (char*)malloc(buf_size * BUF_MULT);
            memcpy(new_buf, buf, offset);
            free(buf);
            buf = new_buf;
            buf_size *= BUF_MULT;
        }
    }

    // Return points for data
    *inp_len = offset;
    *inp = buf;
    return 0;
}


// Performs a single pass over the data to hash it
__global__ void hash_data(int n, char *in, char *out) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x);
    if (offset < n) {
        MurmurHash3_x64_128(in + (offset * PROCESS_LINE_WIDTH), INPUT_LINE_WIDTH, 0, out + (offset * HASH_WIDTH));
    }
}


// Performs a single pass over the data to extract
// the bucket and position of each element
__global__ void extract_hll(int n, char *in, char *out) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x);
    if (offset < n) {
        uint64_t *hash = (uint64_t*)(in + (HASH_WIDTH * offset));

        // Get the first HLL_PREFIX_BITS to determine the bucket
        int bucket = hash[0] >> (64 - HLL_PREFIX_BITS);

        // Finds the position of the least significant 1 (0 to 64)
        int position = __ffsll(hash[1]);

        // Adjust for the limit of the bucket
        if (position == 0) {
            position = HLL_MAX_SCAN - 1;
        } else
            position = min(position, HLL_MAX_SCAN) - 1;

        // Update the output
        uint16_t *outp = ((uint16_t*)out) + offset;
        *outp = ((bucket << HLL_BUCKET_WIDTH) | position);
    }
}


// Uses a two dimensional grid to build the HLL
__global__ void build_hll(int n, uint16_t *in, unsigned int *out) {
    __shared__ int maxPos;
    maxPos = 0;
    __syncthreads();
    int offset = (blockIdx.y * blockDim.y + blockIdx.x * blockDim.x + threadIdx.x);

    if (offset < n) {
        // Extract the parts
        uint16_t val = *(in + offset);
        int bucket = val >> HLL_BUCKET_WIDTH;

        // Only continue if the bucket matches the y index
        if (bucket != blockIdx.y) return;

        // Update the maximum position
        int pos = val & ((1 << HLL_BUCKET_WIDTH) - 1);
        atomicMax(&maxPos, pos);

        // Wait for all the maximums to be sync'd
        __syncthreads();
        atomicMax(&out[blockIdx.y], maxPos);
    }
}


__host__ int main(int argc, char **argv) {
    // Read the input
    printf("Reading input...\n");
    char *inp;
    int inp_len;
    if (read_input(&inp, &inp_len) || inp_len == 0)
        return 1;

    // Move the data to the GPU
    printf("Copying to GPU...\n");
    char *gpu_in, *hashed;
    cudaMalloc((void**)&gpu_in, inp_len);
    cudaMemcpy(gpu_in, inp, inp_len, cudaMemcpyHostToDevice);

    // Determine block sets
    int n = inp_len / PROCESS_LINE_WIDTH;
    cudaMalloc((void**)&hashed, HASH_WIDTH * n);
    int blocks = ceil((float)n / (float)THREADS_PER_BLOCK);

    // Hash all the data for the HLL construction
    printf("Hashing data... (%d lines, %d blocks, %d threads)\n", n, blocks, THREADS_PER_BLOCK);
    hash_data<<<blocks, THREADS_PER_BLOCK>>>(n, gpu_in, hashed);
    cudaError_t res = cudaDeviceSynchronize();
    if (res != cudaSuccess) {
        printf("Hashing failed: %s\n", cudaGetErrorString(res));
        return 1;
    }

    // Extract the HLL's values
    printf("Extracting HLL values...\n");
    char *hll_vals;
    cudaMalloc((void**)&hll_vals, n * 2);
    extract_hll<<<blocks, THREADS_PER_BLOCK>>>(n, hashed, hll_vals);
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) {
        printf("HLL extraction failed: %s\n", cudaGetErrorString(res));
        return 1;
    }

    // Build the HLL's
    printf("Building HLL...\n");
    int hll_size = HLL_BUCKETS * sizeof(unsigned int);
    unsigned int *hll;
    unsigned int *host_hll = (unsigned int*)malloc(hll_size);
    memset(host_hll, 0, hll_size);
    cudaMalloc((void**)&hll, hll_size);
    cudaMemcpy(hll, host_hll, hll_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(blocks, HLL_BUCKETS);
    build_hll<<<dimGrid, THREADS_PER_BLOCK>>>(n, (uint16_t*)hll_vals, hll);
    res = cudaDeviceSynchronize();
    if (res != cudaSuccess) {
        printf("HLL construction failed: %s\n", cudaGetErrorString(res));
        return 1;
    }

    // Copy the HLL back
    cudaMemcpy(host_hll, hll, hll_size, cudaMemcpyDeviceToHost);

    // Estimate cardinality
    printf("Estimating cardinality...\n");
    double raw = raw_estimate(host_hll);
    double adj = range_corrected(raw, host_hll);
    printf("Est: %0.1f Raw: %0.1f\n", adj, raw);

    // Cleanup
    printf("Cleanup...\n");
    free(inp);
    cudaFree(gpu_in);
    cudaFree(hashed);

    return 0;
}

