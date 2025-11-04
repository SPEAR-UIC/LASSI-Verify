#!/bin/bash

# Ensure CUDA library paths are set
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Define paths
SRC_DIR="/home/cc/lassi/LASSI_dev/source_codes/altis/src"
BUILD_DIR="/home/cc/lassi/LASSI_dev/source_codes/altis/build"
CUDA_BIN="/usr/local/cuda/bin"
CXX_BIN="/usr/bin/c++"

# Compile bfs.cu
$CUDA_BIN/nvcc -forward-unknown-to-host-compiler \
  --options-file $BUILD_DIR/src/cuda/level1/bfs/CMakeFiles/bfsLib.dir/includes_CUDA.rsp \
  -std=c++11 \
  "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
  -w -MD \
  -MT $BUILD_DIR/src/cuda/level1/bfs/CMakeFiles/bfsLib.dir/bfs.cu.o \
  -MF $BUILD_DIR/src/cuda/level1/bfs/CMakeFiles/bfsLib.dir/bfs.cu.o.d \
  -x cu \
  -c $SRC_DIR/cuda/level1/bfs/bfs.cu \
  -o $BUILD_DIR/src/cuda/level1/bfs/CMakeFiles/bfsLib.dir/bfs.cu.o

# Create the static library libbfsLib.a
cd $BUILD_DIR/src/cuda/level1/bfs
/usr/bin/ar qc libbfsLib.a CMakeFiles/bfsLib.dir/bfs.cu.o
/usr/bin/ranlib libbfsLib.a

# Compile main.cu
$CUDA_BIN/nvcc -forward-unknown-to-host-compiler \
  --options-file $BUILD_DIR/src/cuda/level1/CMakeFiles/bfs.dir/includes_CUDA.rsp \
  -std=c++11 \
  "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
  -w -MD \
  -MT $BUILD_DIR/src/cuda/level1/CMakeFiles/bfs.dir/__/common/main.cu.o \
  -MF $BUILD_DIR/src/cuda/level1/CMakeFiles/bfs.dir/__/common/main.cu.o.d \
  -x cu \
  -c $SRC_DIR/cuda/common/main.cu \
  -o $BUILD_DIR/src/cuda/level1/CMakeFiles/bfs.dir/__/common/main.cu.o

# Link and generate the executable bfs
$CXX_BIN $BUILD_DIR/src/cuda/level1/CMakeFiles/bfs.dir/__/common/main.cu.o \
  -o $BUILD_DIR/bin/level1/bfs \
  -L/usr/local/cuda-12.4/targets/sbsa-linux/lib/stubs \
  -L/usr/local/cuda-12.4/targets/sbsa-linux/lib \
  $BUILD_DIR/src/common/libAltisCommon.a \
  $BUILD_DIR/src/cuda/level1/bfs/libbfsLib.a \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
