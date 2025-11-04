# miniMDock: Build & Run Guide (CUDA + NVIDIA GPU)

---

https://github.com/ORNL-PE/miniMDock

## 1. Environment Setup

Make sure your system has:

- **CUDA Toolkit ≥ 8.0** (Recommended: ≥ 11.0; tested on CUDA 12.6)
- **GCC compiler** (e.g., GCC 9+ or 13+)
- An NVIDIA GPU (e.g., A100)

Then, set the required environment variables:

```bash
export GPU_INCLUDE_PATH=/usr/local/cuda/include
export GPU_LIBRARY_PATH=/usr/local/cuda/lib64
```

For HIP: (check the version number with ```bash ls /opt/rocm* ```)
```bash
export HIP_PLATFORM=amd
export GPU_LIBRARY_PATH=/opt/rocm-6.3.3/lib
export GPU_INCLUDE_PATH=/opt/rocm-6.3.3/include
export HIP_PATH=/opt/rocm-6.3.3
```

Need to modify the makefile for deprecated assignment (because using a new version of ROCm 6.x):
Change:
IFLAGS=-I$(COMMON_DIR) -I$(HOST_INC_DIR) -I$(GPU_INCLUDE_PATH) -I$(HIP_PATH) -D__HIP_PLATFORM_HCC__  -I$(KRNL_DIR)

To:
IFLAGS=-I$(COMMON_DIR) -I$(HOST_INC_DIR) -I$(GPU_INCLUDE_PATH) -I$(HIP_PATH) -D__HIP_PLATFORM_AMD__  -I$(KRNL_DIR)

---

## 2. Add A100 Support to Makefile

Open the `Makefile` and update the architecture list to include A100 (compute capability 8.0):

```make
TARGETS = 52 61 70 80
```

This ensures that the binary supports your GPU (`sm_80`).

---

## 3. Compile miniMDock (CUDA Version)

Clean previous builds and compile with:

```bash
make clean
make API=CUDA DEVICE=GPU CARD=NVIDIA NUMWI=64
```

This will produce a binary such as:

```bash
./bin/autodock_gpu_64wi
```

You may optionally rename it for clarity:

```bash
mv ./bin/autodock_gpu_64wi ./bin/autodock_cuda_nv_64wi
```

more changes for HIP:

1) In host/src/main.cpp, remove the third parameter:
#ifndef USE_KOKKOS
#ifdef USE_OMPT
        cData.devnum = 0;
	setup_gpu_for_docking(cData);
#else
	setup_gpu_for_docking(cData, tData);  // removed mypars [mtd]
#endif
#endif

2) In Makefile.Hip.Amd, add AFTER the HIP_FLAGS settings:

# [mtd] add
# Suppress excessive warnings for ROCm / GCC
HIP_FLAGS += -Wno-pass-failed -w
CFLAGS    += -Wno-unused-result -Wno-format -w

3) In Makefile.Hip.Amd, add no-pie to:
CPP = g++ -no-pie
---

## 4. Run miniMDock

Use the following command to launch docking:

```bash
./bin/autodock_cuda_nv_64wi -lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 100
```

or

./bin/autodock_gpu_64wi -lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 100

### To increase run time:

- Increase the number of runs (e.g., `-nrun 100`)
- Use more complex ligand files from the `input/` directory
- Run multiple ligands sequentially or in batch

Example:

```bash
./bin/autodock_cuda_nv_64wi -lfile ./input/complex_ligand.pdbqt -nrun 100
```

---

## �� Notes

- The ligand file must be in `.pdbqt` format.
- Output logs and runtime messages will be printed to stdout.
- Be sure to check CUDA compatibility if switching systems.

---

## �� Tested System

- **OS:** Ubuntu 24.04  
- **GPU:** NVIDIA A100 (CUDA Compute Capability 8.0)  
- **CUDA Version:** 12.6  
- **GCC Version:** 13.3  

---

## �� Project Structure & File Dependencies

This diagram illustrates the high-level source code structure and function call relationships within `miniMDock`:

![Project Structure](miniMDock_structure.png)

---

## �� License

miniMDock is available under GNU GPL v2 and/or LGPL v2.1.