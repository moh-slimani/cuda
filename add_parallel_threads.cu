#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define N 512

__global__ void add(int *a, int *b, int *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void random_ints(int *a, int size) {
  int i;
  for (i = 0; i < size; ++i)
    a[i] = rand() % 10;
}

void show_ints(int *a, int size) {
  int i;
  for (i = 0; i < size; ++i)
    cout << a[i] << ",";
  cout << "...." << endl;
}

int main() {
  int *a, *b, *c;       // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of
                        // a, b, c
  int size = N * sizeof(int);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Setup input values
  a = (int *)malloc(size);
  random_ints(a, N);
  b = (int *)malloc(size);
  random_ints(b, N);
  c = (int *)malloc(size);

  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N threads
  add<<<1, N>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  show_ints(a, 30);
  show_ints(b, 30);
  cout << "Result : " << endl;
  show_ints(c, 30);

  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
