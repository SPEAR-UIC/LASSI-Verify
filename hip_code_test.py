"""
General HIP Code Test for LASSI
Flexible test for compiling and executing any HIP source code file
Located in top-level directory alongside LASSI_power_pipeline_v3.ipynb

# Quick test from terminal:
python -c "
from hip_code_test import quick_test
quick_test('source_codes/HeCBench/jacobi/jacobi-hip_main.cu')
"
"""

import subprocess
import os
import shlex
import time
# Import LASSI utilities from subfolder
from lassi.gpu_detector import detect_gpu_vendors
from lassi.amd_power_test import AMDPowerMonitor

def test_hip_source_code(source_file_path, 
                        input_parameters=None,
                        compiler_flags=None,
                        timeout=30,
                        measure_power=False):
    """
    Test compilation and execution of any HIP source code file
    
    Args:
        source_file_path: Full path to HIP source file
        input_parameters: List of command line arguments for execution
        compiler_flags: Custom compiler flags (if None, uses LASSI defaults)
        timeout: Execution timeout in seconds
        measure_power: Whether to monitor power during execution
    """
    
    print(f"=== LASSI HIP Code Test: {os.path.basename(source_file_path)} ===")
    
    # Check if source file exists
    if not os.path.exists(source_file_path):
        print(f"❌ Source file not found: {source_file_path}")
        return False
    
    print(f"✅ Source file found: {source_file_path}")
    
    # Get file info
    file_size = os.path.getsize(source_file_path)
    with open(source_file_path, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"   File size: {file_size} bytes")
    print(f"   Lines of code: {line_count}")
    
    # Check GPU availability
    vendors = detect_gpu_vendors()
    amd_available = any(v['vendor'] == 'amd' for v in vendors)
    
    if not amd_available:
        print("❌ No AMD GPUs detected - cannot test HIP code")
        return False
    
    print(f"✅ AMD GPUs available for testing")
    
    # Set default compiler flags (LASSI-style for AMD)
    if compiler_flags is None:
        compiler_flags = "-std=c++14 -Wall -O3 --amdgpu-target=gfx908"
    
    print(f"   Compiler flags: {compiler_flags}")
    
    # Determine output file name
    base_name = os.path.splitext(source_file_path)[0]
    output_file = base_name
    
    # Split compiler flags
    compile_kwds_list = shlex.split(compiler_flags)
    
    # Build compilation command
    compile_command = ['hipcc'] + compile_kwds_list + [source_file_path, '-o', output_file]
    
    print(f"\n--- Compilation Phase ---")
    print(f"Command: {' '.join(compile_command)}")
    
    # Compile
    compile_start = time.time()
    try:
        result = subprocess.run(compile_command, capture_output=True, text=True)
        compile_time = time.time() - compile_start
        
        if result.returncode == 0:
            print(f"✅ Compilation successful! ({compile_time:.2f}s)")
            if result.stderr.strip():  # Show warnings if any
                print(f"   Warnings: {result.stderr.strip()}")
        else:
            print(f"❌ Compilation failed! ({compile_time:.2f}s)")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False
    
    # Check if executable was created
    if not os.path.exists(output_file):
        print(f"❌ Executable not created: {output_file}")
        return False
    
    # Prepare execution
    print(f"\n--- Execution Phase ---")
    
    # Set default input parameters
    if input_parameters is None:
        input_parameters = []
    
    # Build execution command
    execute_command = [f'./{os.path.basename(output_file)}'] + input_parameters
    
    print(f"Command: {' '.join(execute_command)}")
    print(f"Working directory: {os.path.dirname(os.path.abspath(output_file))}")
    
    # Initialize power monitoring if requested
    power_monitor = None
    power_data = []
    if measure_power:
        print("🔋 Power monitoring enabled")
        power_monitor = AMDPowerMonitor()
        if power_monitor.gpu_count == 0:
            print("⚠️  No AMD GPUs for power monitoring, proceeding without")
            measure_power = False
    
    # Change to the directory containing the executable
    original_dir = os.getcwd()
    exec_dir = os.path.dirname(os.path.abspath(output_file))
    if exec_dir:
        os.chdir(exec_dir)
    
    # Execute with optional power monitoring
    exec_start = time.time()
    try:
        if measure_power and power_monitor:
            # Get baseline power
            baseline_power = power_monitor.get_power_usage(0)
            print(f"   Baseline power: {baseline_power:.2f} W")
        
        result = subprocess.run(execute_command, 
                              capture_output=True, 
                              text=True, 
                              timeout=timeout)
        exec_time = time.time() - exec_start
        
        if measure_power and power_monitor:
            # Get power after execution
            final_power = power_monitor.get_power_usage(0)
            print(f"   Final power: {final_power:.2f} W")
        
        if result.returncode == 0:
            print(f"✅ Execution successful! ({exec_time:.2f}s)")
            
            # Show output (truncated if too long)
            stdout = result.stdout.strip()
            if stdout:
                print("--- Program Output ---")
                if len(stdout) > 500:
                    print(stdout[:500] + "\n... (output truncated)")
                else:
                    print(stdout)
            else:
                print("   (No stdout output)")
                
            # Show any stderr (might be informational)
            stderr = result.stderr.strip()
            if stderr:
                print("--- Program Messages ---")
                if len(stderr) > 200:
                    print(stderr[:200] + "\n... (messages truncated)")
                else:
                    print(stderr)
        else:
            print(f"❌ Execution failed! ({exec_time:.2f}s)")
            print(f"   Return code: {result.returncode}")
            if result.stdout.strip():
                print(f"   Stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"   Stderr: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Execution timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    # Clean up executable (optional)
    try:
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"✅ Cleaned up executable: {output_file}")
    except:
        print(f"⚠️  Could not clean up: {output_file}")
    
    print(f"\n✅ HIP Code Test Complete!")
    print(f"   Total time: {compile_time + exec_time:.2f}s")
    return True

# Convenience functions for easy testing
def quick_test(source_file_path, input_params=None):
    """Quick test with default settings"""
    return test_hip_source_code(source_file_path, input_params)

def power_test(source_file_path, input_params=None):
    """Test with power monitoring enabled"""
    return test_hip_source_code(source_file_path, input_params, measure_power=True)

# Example usage function
def example_usage():
    """Show example usage"""
    print("=== HIP Code Test Examples ===")
    print()
    print("# Test jacobi with no parameters:")
    print('test_hip_source_code("source_codes/HeCBench/jacobi/jacobi-hip_main.cpp")')
    print()
    print("# Test with input parameters:")
    print('test_hip_source_code("source_codes/HeCBench/matrix-rotate/matrix-rotate-hip_main.cpp", ["1000", "10"])')
    print()
    print("# Test with custom compiler flags:")
    print('test_hip_source_code("my_code.cpp", [], "-std=c++17 -O2 --amdgpu-target=gfx908")')
    print()
    print("# Quick test:")
    print('quick_test("my_hip_code.cpp", ["param1", "param2"])')
    print()
    print("# Test with power monitoring:")
    print('power_test("source_codes/HeCBench/jacobi/jacobi-hip_main.cpp")')

if __name__ == "__main__":
    example_usage()