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


__global__ void hash_data(int n, char *in, char *out) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x);
    if (offset < n) {
        printf("Offset: %d (%d)\n", offset, blockIdx.x);
        MurmurHash3_x64_128(in + (offset * PROCESS_LINE_WIDTH), INPUT_LINE_WIDTH, 0, out + (offset * HASH_WIDTH));
    }
}


__host__ int main(int argc, char **argv) {
    // Read the input
    printf("Reading input...\n");
    char *inp;
    int inp_len;
    if (read_input(&inp, &inp_len) || inp_len == 0)
        return 1;

    // Print input bytes
    printf("Read %d bytes\n\n", inp_len);

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
    printf("Hashing data... (%d lines, %d blocks)\n", n, blocks);
    hash_data<<<blocks, THREADS_PER_BLOCK>>>(n, gpu_in, hashed);
    cudaError_t res = cudaDeviceSynchronize();
    if (res != cudaSuccess) {
        printf("Hashing failed: %s\n", cudaGetErrorString(res));
        return 1;
    }

    // Build the HLL's
    printf("Building HLL...\n");
    // TODO...

    // Estimate cardinality
    printf("Estimating cardinality...\n");
    // TODO...

    // Cleanup
    printf("Cleanup...\n");
    free(inp);
    cudaFree(gpu_in);
    cudaFree(hashed);

    return 0;
}

