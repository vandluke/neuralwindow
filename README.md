# neuralwindow

<p align="center">
    <img src="extra/logo.png", height=300, width=300>
</p>

<div align="center">
  
  [![Build Unit Tests](https://github.com/vandenbergheluke/neuralwindow/actions/workflows/cmake.yml/badge.svg)](https://github.com/vandenbergheluke/neuralwindow/actions/workflows/cmake.yml)
  
</div>

## Description

A basic deep learning library written in C.

## Setup

Clone the repository

```bash
git clone https://github.com/vandenbergheluke/neuralwindow.git
```

Make project the current working directory

```bash
cd neuralwindow 
```

### Ubuntu 22.04 LTS

Update and upgrade Ubuntu

```bash
sudo apt update && sudo apt upgrade
```

Install MKL

```bash
sudo apt-get install -y intel-mkl
```

Install OpenBLAS

```bash
sudo apt-get install -y libopenblas-dev
```

Install MAGMA

```bash
sudo wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.7.2.tar.gz
tar -xvzf magma-2.7.2.tar.gz
cd magma-2.7.2
mkdir build && cd build
cmake -DUSE_FORTRAN=off ..
make
sudo make install
```

Install Check

```bash
sudo apt-get install -y check
```

Install Valgrind

```bash
sudo apt-get install -y valgrind
```

Install Libtorch

```bash
sudo wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-shared-with-deps-2.0.1+cpu.zip
```

Install CUDA Regular (Optional)

```bash
lspci | grep -i nvidia  
sudo apt-get install linux-headers-$(uname -r)  
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
```

Install CUDA WSL (Optional)

```bash
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
export PATH=/usr/local/cuda-12/bin${PATH:+:${PATH}}
```

After CUDA installation you will need to reboot. Verify installation with

```bash
nvcc --version
```

Install OpenMP

```bash
sudo apt install libomp-dev
```

Install Doxygen (Optional)

```bash
sudo apt-get install doxygen
```

Install Graphviz (Optional)

```bash
sudo apt install graphviz
sudo apt-get install -y libgraphviz-dev
```

Install LCOV

```bash
sudo apt-get install lcov
```

Install MathGL

```bash
sudo apt-get install zlib1g-dev
sudo apt-get install libpng-dev
wget -O mathgl-8.0.1.tar.gz "https://gigenet.dl.sourceforge.net/project/mathgl/mathgl/mathgl%208.0/mathgl-8.0.1.tar.gz"
tar -xvzf mathgl-8.0.1.tar.gz
cd mathgl-8.0.1
mkdir build
cd build
cmake ../
make
sudo make install
```

## Build

To build neuralwindow run

```bash
mkdir build
cd build
cmake ..
make
```

To build and test without CUDA define enviroment variable `CPU_ONLY=1`.

```bash
CPU_ONLY=1 cmake ..
```

Display Debug information by defining the enviroment variable `DEBUG=1` and
skipping plotting with `-DPLOT=OFF`.

```bash
DEBUG=1 cmake -DCMAKE_PREFIX_PATH=../libtorch/ -DPLOT=OFF ..
```

Generate graph by defining the enviroment variable `GRAPH=1`.

```bash
GRAPH=1 ./executable
dot -Tpng graph.dot > graph.png
dot -Tsvg graph.dot > graph.svg
```

## Test

Run all the unit tests with

```bash
ctest --output-on-failure
```

A specific test executable can be run with

```bash
./test/<test_name>
```

Valgrind can be run with a specific test exectuable with the command

```bash
make valgrind_<test_name>
```

To generate a Valgrind suppression file run

```bash
valgrind --leak-check=full --show-reachable=yes --error-limit=no --gen-suppressions=all --log-file=suppressions.log ./test/<test_name>
cat ./suppressions.log | ./../parse_valgrind_suppressions.sh > suppressions.supp
```

To generate the coverage pages run

```bash
make report
```

To generate the documentation pages run the following command from the project root

```bash
doxygen Doxyfile
```

## Datasets

Create a data folder in the project directory.

```bash
cd data
```

Download and unzip MNIST

```bash
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip -f -d train-images-idx3-ubyte.gz
gzip -f -d train-labels-idx1-ubyte.gz 
gzip -f -d t10k-images-idx3-ubyte.gz
gzip -f -d t10k-labels-idx1-ubyte.gz
```
