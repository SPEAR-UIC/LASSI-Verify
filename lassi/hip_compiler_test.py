"""
HIP Compilation Test for LASSI
Tests HIP compilation workflow similar to CUDA compilation
Created: September 16, 2025
Matthew T. Dearing
"""

import subprocess
import os
import shlex

def detect_hip_compiler():
    """Check if HIP compiler is available"""
    try:
        result = subprocess.run(['hipcc', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✅ HIP Compiler Available:")
        print(result.stdout.split('\n')[0])  # First line with version
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ HIP Compiler (hipcc) not found")
        return False

def get_hip_architecture():
    """Get HIP architecture for MI100 (should be gfx908)"""
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'gfx' in line.lower() and 'name' in line.lower():
                # Look for architecture name
                if 'gfx908' in line:
                    print("✅ Detected MI100 Architecture: gfx908")
                    return "gfx908"
        
        # Fallback - assume gfx908 for MI100
        print("⚠️  Architecture auto-detection failed, assuming gfx908 for MI100")
        return "gfx908"
    except:
        print("⚠️  Could not detect architecture, assuming gfx908 for MI100")
        return "gfx908"

def compile_hip_code(source_file, compiler_flags=""):
    """Compile HIP code similar to CUDA compilation in LASSI"""
    
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return False, ""
    
    # Determine output file name (similar to LASSI CUDA logic)
    if source_file.endswith('.cu'):
        output_file = source_file.replace('.cu', '')
    elif source_file.endswith('.cpp'):
        output_file = source_file.replace('.cpp', '')
    elif source_file.endswith('.hip'):
        output_file = source_file.replace('.hip', '')
    else:
        output_file = source_file + '_compiled'
    
    # Split compiler flags
    compile_kwds_list = shlex.split(compiler_flags) if compiler_flags else []
    
    # Build compilation command (similar to LASSI compile_code function)
    compile_command = ['hipcc'] + compile_kwds_list + [source_file, '-o', output_file]
    
    print(f"Compile command: {' '.join(compile_command)}")
    
    try:
        # Execute compilation
        result = subprocess.run(compile_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ HIP compilation successful!")
            return True, output_file
        else:
            print("❌ HIP compilation failed:")
            print("STDERR:", result.stderr)
            return False, ""
            
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False, ""

def test_hip_compilation():
    """Test HIP compilation workflow"""
    print("=== LASSI HIP Compilation Test ===")
    
    # Check if HIP compiler is available
    if not detect_hip_compiler():
        return False
    
    # Get architecture
    arch = get_hip_architecture()
    
    # Create a simple test HIP file (similar to our earlier test)
    test_hip_content = '''#include <hip/hip_runtime.h>
#include <iostream>

__global__ void test_kernel() {
    int tid = threadIdx.x;
    printf("Hello from HIP thread %d\\n", tid);
}

int main() {
    std::cout << "Launching HIP kernel..." << std::endl;
    test_kernel<<<1, 4>>>();
    
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "HIP test completed successfully!" << std::endl;
    return 0;
}'''
    
    # Write test file
    test_file = "test_hip_compilation.cpp"
    with open(test_file, 'w') as f:
        f.write(test_hip_content)
    
    print(f"✅ Created test HIP file: {test_file}")
    
    # Test compilation with different flag sets
    test_cases = [
        {
            "name": "Basic compilation",
            "flags": f"-std=c++14 -O3"
        },
        {
            "name": "With architecture",
            "flags": f"-std=c++14 -O3 --amdgpu-target={arch}"
        },
        {
            "name": "LASSI-style flags", 
            "flags": f"-std=c++14 -Wall -O3 --amdgpu-target={arch}"
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_case['name']} ---")
        success, output_file = compile_hip_code(test_file, test_case['flags'])
        
        if success:
            success_count += 1
            print(f"✅ Output: {output_file}")
            
            # Test execution
            try:
                result = subprocess.run([f'./{output_file}'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Execution successful!")
                    print("Output:", result.stdout.strip())
                else:
                    print("⚠️  Compilation OK, but execution failed")
                    print("Error:", result.stderr)
            except Exception as e:
                print(f"⚠️  Could not test execution: {e}")
            
            # Clean up executable
            if os.path.exists(output_file):
                os.remove(output_file)
    
    # Clean up test file
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print(f"\n=== Results: {success_count}/{len(test_cases)} compilation tests passed ===")
    
    if success_count > 0:
        print("✅ HIP compilation workflow ready for LASSI integration!")
        return True
    else:
        print("❌ HIP compilation needs troubleshooting")
        return False

if __name__ == "__main__":
    test_hip_compilation()