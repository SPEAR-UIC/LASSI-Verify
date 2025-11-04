"""
AMD Power Monitoring Test for LASSI
Tests basic AMD GPU power monitoring using rocm-smi
Created: September 16, 2025
Matthew T. Dearing
"""

import subprocess
import time
import re
from threading import Thread
import threading
from queue import Queue

class AMDPowerMonitor:
    def __init__(self):
        self.gpu_count = self._get_gpu_count()
        
    def _get_gpu_count(self):
        """Get number of AMD GPUs"""
        try:
            result = subprocess.run(['rocm-smi', '--showid'], 
                                  capture_output=True, text=True, check=True)
            lines = [line for line in result.stdout.split('\n') if 'GPU' in line and ':' in line]
            return len(lines)
        except:
            return 0
    
    def get_power_usage(self, gpu_id=0):
        """Get current power usage for specified GPU (in Watts)"""
        try:
            if gpu_id is None:
                # Get all GPUs power
                result = subprocess.run(['rocm-smi', '--showpower'], 
                                      capture_output=True, text=True, check=True)
            else:
                # Get specific GPU power using -d flag
                result = subprocess.run(['rocm-smi', '-d', str(gpu_id), '--showpower'], 
                                      capture_output=True, text=True, check=True)
            
            # Parse the output to extract power value
            lines = result.stdout.split('\n')
            for line in lines:
                if f'GPU[{gpu_id}]' in line and 'Average Graphics Package Power' in line:
                    # Extract power value using regex
                    match = re.search(r'(\d+\.?\d*)', line.split(':')[-1])
                    if match:
                        return float(match.group(1))
            
            # Fallback: if specific GPU not found, try parsing any power line
            for line in lines:
                if 'Average Graphics Package Power' in line and 'W' in line:
                    match = re.search(r'(\d+\.?\d*)', line.split(':')[-1])
                    if match:
                        return float(match.group(1))
            
            return 0.0
        except Exception as e:
            print(f"Error getting AMD GPU power: {e}")
            return 0.0
    
    def get_all_gpu_power(self):
        """Get power for all GPUs at once (more efficient)"""
        try:
            result = subprocess.run(['rocm-smi', '--showpower'], 
                                  capture_output=True, text=True, check=True)
            
            powers = {}
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU[' in line and 'Average Graphics Package Power' in line:
                    # Extract GPU ID
                    gpu_match = re.search(r'GPU\[(\d+)\]', line)
                    # Extract power value
                    power_match = re.search(r'(\d+\.?\d*)', line.split(':')[-1])
                    
                    if gpu_match and power_match:
                        gpu_id = int(gpu_match.group(1))
                        power = float(power_match.group(1))
                        powers[gpu_id] = power
            
            return powers
        except Exception as e:
            print(f"Error getting all AMD GPU power: {e}")
            return {}
    
    def get_memory_info(self, gpu_id=0):
        """Get memory usage info"""
        try:
            # Try different memory info types
            result = subprocess.run(['rocm-smi', '-d', str(gpu_id), '--showmeminfo', 'vram'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout
        except:
            try:
                # Fallback to basic memory usage
                result = subprocess.run(['rocm-smi', '-d', str(gpu_id), '--showmemuse'], 
                                      capture_output=True, text=True, check=True)
                return result.stdout
            except:
                return "Memory info unavailable"

def amd_power_collection_thread(power_queue, stop_event, gpu_id=0, interval=0.01):
    """AMD equivalent of NVIDIA power_collection_thread - FIXED VERSION"""
    monitor = AMDPowerMonitor()
    start_time = time.time()
    
    while not stop_event.is_set():
        try:
            current_time = time.time() - start_time
            power = monitor.get_power_usage(gpu_id)
            if power is not None and power > 0:
                power_queue.put((current_time, power))
            time.sleep(interval)
        except Exception as e:
            print(f"Error in AMD power collection thread: {e}")
            break

def test_amd_power_monitoring():
    """Test AMD power monitoring with FIXED rocm-smi commands"""
    print("=== LASSI AMD Power Monitoring Test - FIXED VERSION ===")
    
    monitor = AMDPowerMonitor()
    
    if monitor.gpu_count == 0:
        print("❌ No AMD GPUs detected")
        return False
    
    print(f"✅ Detected {monitor.gpu_count} AMD GPU(s)")
    
    # Test all GPUs power reading at once (most efficient)
    print("\n--- All GPUs Power Reading Test ---")
    all_powers = monitor.get_all_gpu_power()
    for gpu_id, power in all_powers.items():
        print(f"GPU {gpu_id}: {power:.2f} W")
    
    # Test individual power readings
    print("\n--- Individual Power Reading Test ---")
    for gpu_id in range(monitor.gpu_count):
        power = monitor.get_power_usage(gpu_id)
        print(f"GPU {gpu_id}: {power:.2f} W")
    
    # Test memory info
    print("\n--- Memory Info Test ---")
    memory_info = monitor.get_memory_info(0)
    print("GPU 0 Memory Info:")
    print(memory_info[:300] + "..." if len(memory_info) > 300 else memory_info)
    
    # Test threaded power collection (like NVIDIA version)
    print("\n--- Threaded Power Collection Test (5 seconds) ---")
    power_queue = Queue()
    stop_event = threading.Event()
    
    # Start power collection thread
    power_thread = Thread(target=amd_power_collection_thread, 
                         args=(power_queue, stop_event, 0))
    power_thread.start()
    
    # Collect for 5 seconds
    time.sleep(5)
    
    # Stop collection
    stop_event.set()
    power_thread.join()
    
    # Analyze collected data
    power_data = []
    while not power_queue.empty():
        power_data.append(power_queue.get())
    
    if power_data:
        powers = [p[1] for p in power_data]  # Extract power values
        avg_power = sum(powers) / len(powers)
        min_power = min(powers)
        max_power = max(powers)
        
        print(f"✅ Collected {len(power_data)} power samples")
        print(f"   Average: {avg_power:.2f} W")
        print(f"   Range: {min_power:.2f} - {max_power:.2f} W")
        print(f"   Sample rate: ~{len(power_data)/5:.1f} Hz")
        
        if avg_power > 0:
            print("✅ Power monitoring working correctly!")
        else:
            print("⚠️  Power readings are zero - check GPU activity")
    else:
        print("❌ No power data collected")
        return False
    
    print("\n✅ AMD Power Monitoring Test Complete!")
    return True

if __name__ == "__main__":
    test_amd_power_monitoring()