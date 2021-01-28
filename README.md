# CUDA installation

The Nvidia CUDA toolkit is an extension of the GPU parallel computing platform and programming model. The Nvidia CUDA installation consists of inclusion of the official Nvidia CUDA repository followed by the installation of relevant meta package and configuring path the the executable CUDA binaries. 

## Software Requirements and Conventions Used

| Category | Requirements, Conventions or Software Version Used |
|----------|----------------------------------------------------|
| System   | Ubuntu 20.04   |
| Hardware | Nvidia Graphic Card   |
| Software | - Nvidia Drivers <br> - CUDA |
| Other    | Privileged access to your Linux system as root or via the sudo command. |
|Conventions| **#** - requires given *linux commands* to be executed with root privileges either directly as a root user or by use of `sudo` command <br> **$** - requires given *linux commands* to be executed as a regular non-privileged user  |

## Nvidia Drivers installation

**Step 1** : First, detect the model of your nvidia graphic card and the recommended driver. To do so execute the following command. Please note that your output and recommended driver will most likely be different: 

```text
$ ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:1c.0/0000:01:00.0 ==
modalias : pci:v000010DEd00001D10sv00001025sd0000119Abc03sc02i00
vendor   : NVIDIA Corporation
model    : GP108M [GeForce MX150]
driver   : nvidia-driver-460 - third-party non-free recommended
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-450 - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-390 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin

```

From the above output we can conclude that the current system has **NVIDIA GeForce MX150** graphic card installed and the recommend driver to install is **nvidia-driver-460**. 

**Step 2** :  use the ubuntu-drivers command again to install all recommended drivers: 

```bash
$ sudo ubuntu-drivers autoinstall
```

**Step 3** : Once the installation is concluded, reboot your system and you are done.

```bash
$ reboot
```

## CUDA installation

**Step 1** Although you might not end up witht he latest CUDA toolkit version, the easiest way to install CUDA on Ubuntu 20.04 is to perform the installation from Ubuntu's standard repositories.

To install CUDA execute the following commands: 

```bash
$ sudo apt update
$ sudo apt install nvidia-cuda-toolkit
```

**Step 2** All should be ready now. Check your CUDA version: 
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```
and to check the Nvidia drivers

```bash
$ nvidia-smi
Thu Jan 28 13:21:12 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.39       Driver Version: 460.39       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce MX150       Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   51C    P3    N/A /  N/A |    361MiB /  2002MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1159      G   /usr/lib/xorg/Xorg                 45MiB |
|    0   N/A  N/A      2173      G   /usr/lib/xorg/Xorg                124MiB |
|    0   N/A  N/A      2357      G   /usr/bin/gnome-shell              133MiB |
|    0   N/A  N/A      5678      G   ...AAAAAAAAA= --shared-files       50MiB |
+-----------------------------------------------------------------------------+
```

**commun problem**
sometimes the default configurtion for the nvidia drivers is not setup properly to fix this i used the following commands

```bash
$ sudo mv /lib/modprobe.d/blacklist-nvidia.conf /lib/modprobe.d/blacklist-nvidia.conf.back
$ sudo update-initramfs -u
$ reboot
```


## Compile a Sample CUDA code

Confirm the installation by compiling an example CUDA C code. Save the following code into a file named eg. hello.cu: 

```c
#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
```

 Next, use nvcc the Nvidia CUDA compiler to compile the code and run the newly compiled binary: 

 ```bash
$ nvcc -o hello hello.cu 
$ ./hello 
Max error: 0.000000
```

## Refrences

 -[How to install CUDA on Ubuntu 20.04 Focal Fossa Linux](https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux)
 -[How to install the NVIDIA drivers on Ubuntu 20.04 Focal Fossa Linux](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
 -[An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
 -[Bong-Soo Sohn](http://cau.ac.kr/~bongbong/cg14/)
