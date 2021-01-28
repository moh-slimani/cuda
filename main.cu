#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>

using namespace std;

__global__ void mykernel(void) {}

int main() {
  mykernel<<<1,1>>>();
  cout << "hellow world" << endl;
  return 0;
}

