"""
AMD Power Monitoring Integration for LASSI
AMD equivalent functions to match existing NVIDIA power monitoring
Created: September 16, 2025
Matthew T. Dearing
"""

import subprocess
import time
import csv
import re
import threading
from threading import Thread
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
        """Get current power usage for specified GPU (in Watts) - AMD equivalent of nvmlDeviceGetPowerUsage"""
        try:
            result = subprocess.run(['rocm-smi', '-d', str(gpu_id), '--showpower'], 
                                  capture_output=True, text=True, check=True)
            
            lines = result.stdout.split('\n')
            for line in lines:
                if f'GPU[{gpu_id}]' in line and 'Average Graphics Package Power' in line:
                    match = re.search(r'(\d+\.?\d*)', line.split(':')[-1])
                    if match:
                        return float(match.group(1))
            return 0.0
        except:
            return 0.0

# AMD equivalent of NVIDIA power monitoring functions
def amd_report_gpu_memory():
    """AMD equivalent of report_gpu_memory()"""
    try:
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                              capture_output=True, text=True, check=True)
        print("\n----- AMD GPU Memory Status:")
        print(result.stdout)
        
        # Also show power and clocks
        result = subprocess.run(['rocm-smi'], 
                              capture_output=True, text=True, check=True)
        print("----- AMD GPU State:")
        print(result.stdout)
    except Exception as e:
        print(f"Failed to query AMD GPU state: {e}")

def amd_power_collection_thread(power_queue, stop_event, gpu_id=0, interval=0.01):
    """AMD equivalent of power_collection_thread() - matches NVIDIA signature exactly"""
    monitor = AMDPowerMonitor()
    start_time = time.time()
    
    while not stop_event.is_set():
        try:
            current_time = time.time() - start_time
            # Get power in Watts (AMD rocm-smi returns W, NVIDIA returns mW)
            power = monitor.get_power_usage(gpu_id)
            if power is not None and power > 0:
                power_queue.put((current_time, power))
            time.sleep(interval)
        except Exception as e:
            print(f"Error in AMD power collection thread: {e}")
            break

def amd_idlepower(experiment_data_directory, trial_number, duration):
    """AMD equivalent of idlepower() function"""
    monitor = AMDPowerMonitor()
    
    start_power_time = time.time()
    power_queue = Queue()
    stop_power_event = threading.Event()
    
    # Start power collection thread
    power_thread = Thread(target=amd_power_collection_thread, args=(power_queue, stop_power_event, 0))
    power_thread.start()
         
    print("Duration for idle time measurement: " + str(round(duration, 2)))
    time.sleep(duration)
    end_power_time = time.time()
    
    stop_power_event.set()
    power_thread.join()
    
    # Import the power_collect_data function from get_power
    from lassi.get_power import power_collect_data
    
    zero_shot = False
    total_energy_joules, ave_power, std_power, metrics_results_string = power_collect_data(
        experiment_data_directory, "idle_power", zero_shot, trial_number, 
        power_queue, "", "", 0, end_power_time - start_power_time, 
        ave_idle_power=0, idle=True)

    return round(total_energy_joules, 2), round(ave_power, 2), round(std_power, 2)