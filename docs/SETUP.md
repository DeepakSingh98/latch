# Environment setup

The release was tested on Ubuntu 24.04, Python 3.12, GCC 13, CUDA Toolkit 12.8,
PyTorch 2.9.1+cu128 and an H100 80 GB with NVIDIA driver 580.178.04.
Start with an H100 machine whose NVIDIA driver is already working.

On Ubuntu 24.04 x86_64, install the build tools and CUDA 12.8 toolkit:

```sh
sudo apt-get update
sudo apt-get install -y build-essential python3.12-venv python3.12-dev wget
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

This installs the toolkit without replacing the driver. Use
[NVIDIA's installation guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)
for other Linux distributions. Keep the two `export` lines in each build shell.

Check `nvidia-smi` on the GPU machine. Its reported CUDA version describes driver
compatibility; `nvcc --version` tells you which toolkit you will compile with.
Then follow the [README](../README.md) to install PyTorch and Latch.

Compilation can run on a CPU-only machine. TinyMoE's fresh build used 16 vCPUs
and 128 GB RAM; the MLP is much smaller. Copy the output and cache directories to
the GPU machine at the same absolute paths, and use the same Python packages.

On first use, Latch also builds a small native binding check with the host C++
compiler and PyTorch headers. It falls back to a Python check if that build fails.
Having a compiler on the GPU host avoids this slower fallback.
