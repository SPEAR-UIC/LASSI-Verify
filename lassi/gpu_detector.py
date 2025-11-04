"""
GPU Vendor Detection Utility for LASSI
Detects available NVIDIA and AMD GPUs.
Created: September 16, 2025
Matthew T. Dearing
"""

import subprocess
import os

def detect_gpu_vendors():
    """Detect all available GPU vendors on the system"""
    vendors = []
    
    # Check for NVIDIA GPUs
    try:
        result = subprocess.run(['nvidia-smi', '-L'], 
                              capture_output=True, text=True, check=True)
        if 'GPU' in result.stdout:
            nvidia_count = len([line for line in result.stdout.split('\n') if 'GPU' in line and ':' in line])
            vendors.append({'vendor': 'nvidia', 'count': nvidia_count, 'tool': 'nvidia-smi'})
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for AMD GPUs  
    try:
        result = subprocess.run(['rocm-smi', '--showid'], 
                              capture_output=True, text=True, check=True)
        if 'GPU' in result.stdout:
            amd_count = len([line for line in result.stdout.split('\n') if 'GPU' in line and ':' in line])
            vendors.append({'vendor': 'amd', 'count': amd_count, 'tool': 'rocm-smi'})
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
        
    return vendors

def test_gpu_detection():
    """Test function to verify GPU detection works"""
    print("=== LASSI GPU Vendor Detection Test ===")
    
    vendors = detect_gpu_vendors()
    
    if not vendors:
        print("❌ No GPUs detected")
        return False
    
    for vendor_info in vendors:
        vendor = vendor_info['vendor']
        count = vendor_info['count']
        tool = vendor_info['tool']
        print(f"✅ {vendor.upper()}: {count} GPU(s) detected via {tool}")
    
    return True

if __name__ == "__main__":
    test_gpu_detection()