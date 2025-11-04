import subprocess
import time
import shutil

try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, \
        nvmlDeviceGetClockInfo, nvmlDeviceGetPerformanceState, NVML_CLOCK_GRAPHICS, NVML_CLOCK_SM
    NVML_AVAILABLE = True
except ImportError:
    print("pynvml not found. Please install it with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    print("Numba not found. CUDA reset will be skipped. Install with: pip install numba")
    NUMBA_AVAILABLE = False

def query_gpu_nvml():
    if not NVML_AVAILABLE:
        return

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        clocks_gr = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
        clocks_sm = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
        power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts to watts
        pstate = nvmlDeviceGetPerformanceState(handle)

        print("\n🔍 GPU State (via NVML):")
        print(f"  Graphics Clock: {clocks_gr} MHz")
        print(f"  SM Clock:       {clocks_sm} MHz")
        print(f"  Power Draw:     {power_w:.2f} W")
        print(f"  Performance State: P{pstate}")

        # Basic idle-state check
        if pstate == 0 or power_w > 40:
            print("\n⚠️ GPU is not fully idle. You may have a lingering context or process.")
        else:
            print("\n✅ GPU appears idle.")

        nvmlShutdown()

    except Exception as e:
        print(f"Error querying NVML: {e}")

def try_cuda_device_reset():
    if not NUMBA_AVAILABLE:
        return

    print("\n⚙️ Attempting cudaDeviceReset via Numba...")
    try:
        cuda.select_device(0)
        cuda.close()
        print("✅ cudaDeviceReset issued successfully.")
    except Exception as e:
        print(f"❌ Failed to reset CUDA device: {e}")

def check_nvidia_smi():
    if not shutil.which("nvidia-smi"):
        print("nvidia-smi not found in PATH.")
        return

    print("\n📋 nvidia-smi query output:")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.gr,clocks.sm,power.draw,pstate",
                "--format=csv"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e.stderr}")

if __name__ == "__main__":
    print("=== GPU State Checker ===")
    query_gpu_nvml()
    check_nvidia_smi()

    response = input("\nWould you like to try cudaDeviceReset via Numba? [y/N] ").strip().lower()
    if response == 'y':
        try_cuda_device_reset()
        print("\n🕑 Waiting 5 seconds before re-checking GPU state...")
        time.sleep(5)
        query_gpu_nvml()
