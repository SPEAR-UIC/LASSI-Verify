#!/bin/bash

# Ensure CUDA library paths are set
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Define paths
SRC_DIR="/home/cc/lassi/LASSI_dev/source_codes/altis/src"
BUILD_DIR="/home/cc/lassi/LASSI_dev/source_codes/altis/build"
CUDA_BIN="/usr/local/cuda/bin"
CXX_BIN="/usr/bin/c++"

# Ensure the build directory exists
mkdir -p $BUILD_DIR

# Create the static library libraytracingLib.a (if not already built)
cd $BUILD_DIR/src/cuda/level2/raytracing/raytracing_cuda
/usr/bin/ar qc libraytracingLib.a
/usr/bin/ranlib libraytracingLib.a

# Compile main.cu
$CUDA_BIN/nvcc -forward-unknown-to-host-compiler \
  --options-file $BUILD_DIR/src/cuda/level2/CMakeFiles/raytracing.dir/includes_CUDA.rsp \
  -std=c++11 \
  "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
  -w -MD \
  -MT $BUILD_DIR/src/cuda/level2/CMakeFiles/raytracing.dir/__/common/main.cu.o \
  -MF $BUILD_DIR/src/cuda/level2/CMakeFiles/raytracing.dir/__/common/main.cu.o.d \
  -x cu \
  -c $SRC_DIR/cuda/common/main.cu \
  -o $BUILD_DIR/src/cuda/level2/CMakeFiles/raytracing.dir/__/common/main.cu.o

# Link and generate the executable raytracing
$CXX_BIN $BUILD_DIR/src/cuda/level2/CMakeFiles/raytracing.dir/__/common/main.cu.o \
  -o $BUILD_DIR/bin/level2/raytracing \
  -L/usr/local/cuda-12.4/targets/sbsa-linux/lib/stubs \
  -L/usr/local/cuda-12.4/targets/sbsa-linux/lib \
  $BUILD_DIR/src/common/libAltisCommon.a \
  $BUILD_DIR/src/cuda/level2/raytracing/raytracing_cuda/libraytracingLib.a \
  -lm -lcudadevrt -lcudart_static -lrt -lpthread -ldl
