#!/bin/bash

# Define paths
SRC_DIR="/home/cc/lassi/LASSI_dev/source_codes/altis/src"
BUILD_DIR="/home/cc/lassi/LASSI_dev/source_codes/altis/build"
CUDA_BIN="/usr/local/cuda/bin"
CXX_BIN="/usr/bin/c++"

# Compile kmmain.cu
$CUDA_BIN/nvcc -forward-unknown-to-host-compiler \
  --options-file $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/includes_CUDA.rsp \
  -std=c++11 \
  "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
  -w -MD \
  -MT $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/kmmain.cu.o \
  -MF $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/kmmain.cu.o.d \
  -x cu \
  -c $SRC_DIR/cuda/level2/kmeans/kmmain.cu \
  -o $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/kmmain.cu.o

# Compile kmeansraw.cu
$CUDA_BIN/nvcc -forward-unknown-to-host-compiler \
  --options-file $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/includes_CUDA.rsp \
  -std=c++11 \
  "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
  -w -MD \
  -MT $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/kmeansraw.cu.o \
  -MF $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/kmeansraw.cu.o.d \
  -x cu \
  -c $SRC_DIR/cuda/level2/kmeans/kmeansraw.cu \
  -o $BUILD_DIR/src/cuda/level2/kmeans/CMakeFiles/kmeansLib.dir/kmeansraw.cu.o

# Create the static library libkmeansLib.a
cd $BUILD_DIR/src/cuda/level2/kmeans
/usr/bin/ar qc libkmeansLib.a CMakeFiles/kmeansLib.dir/kmmain.cu.o CMakeFiles/kmeansLib.dir/kmeansraw.cu.o
/usr/bin/ranlib libkmeansLib.a

# Compile main.cu
$CUDA_BIN/nvcc -forward-unknown-to-host-compiler \
  --options-file $BUILD_DIR/src/cuda/level2/CMakeFiles/kmeans.dir/includes_CUDA.rsp \
  -std=c++11 \
  "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
  -w -MD \
  -MT $BUILD_DIR/src/cuda/level2/CMakeFiles/kmeans.dir/__/common/main.cu.o \
  -MF $BUILD_DIR/src/cuda/level2/CMakeFiles/kmeans.dir/__/common/main.cu.o.d \
  -x cu \
  -c $SRC_DIR/cuda/common/main.cu \
  -o $BUILD_DIR/src/cuda/level2/CMakeFiles/kmeans.dir/__/common/main.cu.o

# Link and generate the executable kmeans
$CXX_BIN $BUILD_DIR/src/cuda/level2/CMakeFiles/kmeans.dir/__/common/main.cu.o \
  -o $SRC_DIR/cuda/level2/kmeans \
  -L/usr/local/cuda-12.4/targets/sbsa-linux/lib/stubs \
  -L/usr/local/cuda-12.4/targets/sbsa-linux/lib \
  $BUILD_DIR/src/common/libAltisCommon.a \
  $BUILD_DIR/src/cuda/level2/kmeans/libkmeansLib.a \
  -lm -lcudadevrt -lcudart_static -lrt -lpthread -ldl
