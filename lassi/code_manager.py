import os
import subprocess
import time
import shlex  # for splitting command line arguments
from tqdm import tqdm  # for progress bar feature
# Used for power consumption
import csv
from threading import Thread
import threading
from queue import Queue
import queue
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
import lassi.get_power
from lassi.format_colors import color

# AMD vendor GPU support imports
from lassi.amd_power import AMDPowerMonitor, amd_power_collection_thread, amd_report_gpu_memory
from lassi.gpu_detector import detect_gpu_vendors



def execute_code_selectvendor(experiment_data_directory, code_version, zero_shot, trial_number, 
                            directory, executable_file, exe_input_parameters, power_measure, 
                            code_source, target_lang, build_script, make_script, 
                            gpu_vendor="auto", wait=10):
    """
    Multi-vendor execute_code with automatic GPU vendor detection
    Supports NVIDIA and AMD devices.
    """
    
    # Auto-detect GPU vendor if not specified
    if gpu_vendor == "auto":
        vendors = detect_gpu_vendors()
        if vendors:
            gpu_vendor = vendors[0]['vendor']  # Use first detected vendor
            print(f"🔍 Using {gpu_vendor.upper()} GPU for execution and power monitoring")
        else:
            print("⚠️  No GPUs detected, falling back to NVIDIA-based execution")
            # If no GPUs detected, fall back to original function
            return execute_code(experiment_data_directory, code_version, zero_shot, trial_number, 
                              directory, executable_file, exe_input_parameters, power_measure, 
                              code_source, target_lang, build_script, make_script, wait)
    
    # For NVIDIA GPUs, use NVIDIA-based function
    if gpu_vendor == "nvidia":
        return execute_code(experiment_data_directory, code_version, zero_shot, trial_number, 
                          directory, executable_file, exe_input_parameters, power_measure, 
                          code_source, target_lang, build_script, make_script, wait)
    
    # For AMD GPUs, implement AMD-specific logic
    elif gpu_vendor == "amd":
        return execute_code_amd(experiment_data_directory, code_version, zero_shot, trial_number, 
                               directory, executable_file, exe_input_parameters, power_measure, 
                               code_source, target_lang, build_script, make_script, wait)
    
    else:
        # Fallback to original function
        print(f"⚠️  Unknown GPU vendor '{gpu_vendor}', falling back to NVIDIA-based execution")
        return execute_code(experiment_data_directory, code_version, zero_shot, trial_number, 
                          directory, executable_file, exe_input_parameters, power_measure, 
                          code_source, target_lang, build_script, make_script, wait)


def make_script_code(folder, code_compiler_kwds, source_lang):
    print("Make - current dir: " + str(os.getcwd()))
    
    # split your single-string of flags into a list
    cmd = ['make'] + code_compiler_kwds.split()

    # build the absolute path to your target folder
    if folder in str(os.getcwd()):
        target_dir = os.getcwd()
    else:
        target_dir = os.path.join(os.getcwd(), folder)
    
    # copy current env and inject your exports
    env = os.environ.copy()
    if source_lang == "CUDA":
        env['GPU_INCLUDE_PATH']  = '/usr/local/cuda/include'
        env['GPU_LIBRARY_PATH']  = '/usr/local/cuda/lib64'

    # disable all warnings in gcc/g++ (and NVCC’s host compiler)
    # env['CFLAGS'] = env.get('CFLAGS', '') + ' -w'
    # env['CXXFLAGS'] = env.get('CXXFLAGS', '') + ' -w'
    # if you’re using nvcc directly in your Makefile you may need:
    # env['NVCCFLAGS'] = env.get('NVCCFLAGS', '') + ' -w'
    
    # Execute the Make file script command
    # make_script_file = str(os.getcwd()) + "/" + folder + "/" + "make"
    #subprocess.run(['chmod', '+x', build_script_file], check=True)
    # print("make_script_file: ", make_script_file)
    result = subprocess.run(
        cmd,
        cwd=target_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False  # True if you want a CalledProcessError on failure
    )

    # Debug printouts
    print(f"Running: {' '.join(cmd)} in {target_dir}")
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    return_result = ""
    if result.returncode == 0:
        print("\n" + color.BOLD + "Make build/Compilation of code successful!\n" + color.END)
    else:
        return_result = result.stderr

    return return_result
        


def build_script_code(folder, code_build_script):
    print("Build script - current dir: " + str(os.getcwd()))
    # Execute the build script command
    build_script_file = str(os.getcwd()) + "/" + folder + "/altis/" + code_build_script + ".sh"
    subprocess.run(['chmod', '+x', build_script_file], check=True)
    result = subprocess.run([build_script_file], capture_output=True, text=True)

    # Check if the compilation was successful
    return_result = ""
    if result.returncode == 0:
        print("\n" + color.BOLD + "Compilation of code successful!" + color.END)
    else:
        # If there was an error, print the error message
        return_result = result.stderr
        print("\n" + color.BOLD + color.RED + "***** Compilation failed." + color.END)
        print(" Error message: ")
        print(return_result)

    return return_result


def compile_code(source_file, lang, code_compiler, code_compiler_kwds):
       
    # Split the compile keywords into a list of arguments
    compile_kwds_list = shlex.split(code_compiler_kwds)

    if lang == "OMP":
        # The desired name of the compiled program
        output_file = source_file.split('.cpp')[0]
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
    
    elif lang == "CUDA":
        # The desired name of the compiled program
        output_file = source_file.split('.cu')[0]
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
    
    elif lang == "HIP":  # NEW: Add HIP support
        # The desired name of the compiled program
        if source_file.endswith('.cu'):
            output_file = source_file.split('.cu')[0]  # HIP can compile .cu files
        elif source_file.endswith('.cpp'):
            output_file = source_file.split('.cpp')[0]  # HIP can compile .cpp files
        elif source_file.endswith('.hip'):
            output_file = source_file.split('.hip')[0]  # Native HIP files
        else:
            output_file = source_file + '_compiled'  # Fallback
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
        
    print("Compile command: " + str(compile_command))
    
    # Execute the compilation command
    result = subprocess.run(compile_command, capture_output=True, text=True)

    # Check if the compilation was successful
    return_result = ""
    if result.returncode == 0:
        print("\n" + color.BOLD + "Compilation of code successful!\n" + color.END)
    else:
        # If there was an error, print the error message
        return_result = result.stderr
        print("\n" + color.BOLD + color.RED + "***** Compilation failed.\n" + color.END)
        print(" Error message: ")
        print(return_result)
        
    return return_result, output_file


def execute_code(experiment_data_directory, code_version, zero_shot, trial_number, directory, executable_file, exe_input_parameters, power_measure, code_source, target_lang, build_script, make_script, wait=10, execution_timeout=300):

    # Wait 20 seconds for the system to settle down
    # print("...WAITING.")
    # time.sleep(20)
    # print("...DONE WAITING
    if power_measure:
        print("\n----- >>> System check BEFORE exe code...")
        lassi.get_power.report_gpu_memory()

    duration = 0
    start_power_time = 0
    total_energy_joules = 0
    ave_power = 0
    std_power = 0
    metrics_results_string = ""

    # Simulating a progress bar while sleeping for 5 seconds
    for _ in tqdm(range(wait), desc="Waiting for the system to settle down for " + str(wait) + " seconds..."):
        time.sleep(1)

    return_code = ""

    print("executable_file: ", executable_file)
    # Capture current directory to switch back after code execution
    if make_script:
        # build the absolute path to your target folder
        if directory in str(os.getcwd()):
            initial_dir = os.getcwd()
        else:
            initial_dir = os.path.join(os.getcwd(), directory)
    else:
        initial_dir = os.getcwd()
    print("initial_dir: ", initial_dir)
    
    # Extract the compiled program to execute
    directory_path = os.path.dirname(executable_file)
    print("directory_path: ", directory_path)
    executable = os.path.basename(executable_file)
    print("executable: ", executable)

    # Change the working directory to the subdirectory of the executable
    if make_script:
        os.chdir(initial_dir)
        # executable = directory_path + "/" + executable
        # executable = experiment_data_directory
    else:
        # os.chdir(initial_dir + "/" + directory_path)
        os.chdir(directory_path)
        # os.chdir(experiment_data_directory)
        
    # Ensure the binary has execute permissions
    subprocess.run(['chmod', '+x', executable])

    # Build execute command
    if build_script:
        execute_command = ['./' + executable] + exe_input_parameters[0].split(' ')
    elif make_script:
        print("exe_input_parameters: ", exe_input_parameters)
        os.chdir(initial_dir)
        execute_command = ['./' + executable] + exe_input_parameters
    else:
        execute_command = ['./' + executable] + exe_input_parameters
    print("Working directory: " + os.getcwd())
    print("Execute command: " + str(execute_command) + "\n")

    # Execute the binary object with input parameters, if provided

    if power_measure:
        try:
            nvmlInit()
        except NVMLError as err:
            print(f"Failed to initialize NVML: {err}")
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
        # Collect idle power average value before exe
        idle_power_before = []
        for _ in range(200):  # 2 seconds @ 0.01s intervals
            p = nvmlDeviceGetPowerUsage(handle) / 1000.0
            idle_power_before.append(p)
            time.sleep(0.01)
        ave_idle_before = sum(idle_power_before) / len(idle_power_before)
     
    # Ensure the file is open and available before executing
    if os.path.exists(executable):
        print("running: executable - ", executable)
        # Hack for now:
        # if make_script:
        #     executable = "README.md"
        try:
            # Open the file to ensure it is accessible
            with open(executable, 'rb') as code_file:
                # If power measurement is enabled, set it up
                if power_measure:
                    start_power_time = time.time()
                    power_queue = queue.Queue()
                    stop_power_event = threading.Event()
    
                    # Start power collection thread
                    power_thread = Thread(target=lassi.get_power.power_collection_thread, args=(power_queue, stop_power_event, handle))
                    power_thread.start()
                    time.sleep(15)
    
                # Start the process
                start_exe_time = time.time() - start_power_time
                repeat = 1
                stdout_lines = []
                stderr_lines = []
                for i in range(0, repeat):
                    process = subprocess.Popen(execute_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    exception_error = ""
                    
                    # Print the output in real-time
                    print("\n----- < BEGIN: STANDARD OUTPUT FROM CODE EXECUTION > -----\n")
                    try:
                        # Read stdout in a separate thread or in a non-blocking way
                        ## while True:
                        ##     line = process.stdout.readline()
                        ##     if not line:
                        ##         break
                        ##     print(line, end='')  # Print each line as it is received
                        ##     stdout_lines.append(line)
                        ## process.stdout.close()
                
                        # Read stderr in a separate thread or in a non-blocking way
                        ## while True:
                        ##     line = process.stderr.readline()
                        ##     if not line:
                        ##         break
                        ##     stderr_lines.append(line)
                        ## process.stderr.close()
                    
                        # Wait for the process to complete and get the return code
                        ## return_code = process.wait()

                        # Use communicate with timeout
                        stdout_output, stderr_output = process.communicate(timeout=execution_timeout)
                        
                        # Print the captured output
                        print(stdout_output, end='')
                        stdout_lines.append(stdout_output)
                        stderr_lines.append(stderr_output)
                        
                        return_code = process.returncode
                        
                        if return_code != 0:
                            ## error_output = ''.join(stderr_lines)
                            error_output = stderr_output
                            print("Program Execution Error:", error_output)
    
                    except subprocess.TimeoutExpired:
                        print(f"\n***** TIMEOUT: Process exceeded {execution_timeout} seconds *****")
                        process.kill()  # Kill the process
                        process.wait()  # Wait for it to terminate
                        return_code = -999  # Custom return code for timeout
                        exception_error = f"Execution timeout after {execution_timeout} seconds"
                        
                        # Try to get any partial output
                        try:
                            stdout_output, stderr_output = process.communicate(timeout=5)
                            stdout_lines.append(stdout_output)
                            stderr_lines.append(stderr_output)
                        except:
                            pass
            
                    except Exception as e:
                        print("An error occurred while executing the program: ", e)
                        exception_error = str(e)
                        return_code = -998  # Custom return code for other exceptions
                        process.kill()
                        process.wait()

                    print("\n----- < END: Standard Output > -----\n")

                total_energy_joules = 0
                ave_power = 0
                
                # Only continue power measurements if no exe error
                if power_measure and (return_code == 0 or any("RESULTS" in s for s in stdout_lines)):           
                    end_exe_time = time.time() - start_power_time
                    duration = end_exe_time - start_exe_time
                    # Continue collecting power data for a short time after inference
                    time.sleep(15)
                    end_power_time = time.time()            
                    # while (end_power_time - start_power_time) < 10:
                        # time.sleep(0.01)
                    #    end_power_time = time.time()
                    stop_power_event.set()
                    power_thread.join()
                    # Collect idle power average value after exe
                    idle_power_after = []
                    for _ in range(200):  # 2 seconds @ 0.01s intervals
                        p = nvmlDeviceGetPowerUsage(handle) / 1000.0
                        idle_power_after.append(p)
                        time.sleep(0.01)
                    ave_idle_after = sum(idle_power_after) / len(idle_power_after)

                    ave_idle_power = (ave_idle_before + ave_idle_after) / 2
        
                    total_energy_joules, ave_power, std_power, metrics_results_string = lassi.get_power.power_collect_data(
                        experiment_data_directory,
                        code_version,
                        zero_shot,
                        trial_number,
                        power_queue, 
                        code_source, 
                        target_lang, 
                        start_exe_time, 
                        end_exe_time,
                        ave_idle_power,
                        idle=False
                    )

                    # Shutdown NVML
                    try:
                        nvmlShutdown()
                    except NVMLError as err:
                        print(f"Failed to shutdown NVML: {err}")
        
        except IOError as e:
            print(f"Pipeline error | An I/O error occurred: {e.strerror}")
        except Exception as e:
            print(f"Pipeline error | An error occurred: {e}")
    else:
        print(f"Pipeline error | The file {executable} does not exist.")
    
    
    # Check if the program ran successfully
    return_result = "0"  # default OK
    if any("RESULTS" in s for s in stdout_lines) or return_code == 0:
        print("\n" + color.BOLD + "Program executed successfully!" + color.END)
    else:
        # If there was an error during execution, print the error message
        return_result = "Program exited with return code: " + str(return_code) + " "
        if return_code == -11:
            return_result += "Segmentation fault detected."
        stderr_output = ''.join(stderr_lines)
        if exception_error != "":
            return_result += "\nException Error: " + exception_error
        if stderr_output:
            return_result += "\nStandard Error: " + stderr_output
        print("\n" + color.BOLD + color.RED + "***** Execution failed." + color.END)
        print("Error message: ")
        print(return_result)
    
    os.chdir(initial_dir)

    stdout_output = ''.join(stdout_lines)

    if power_measure:
        print("\n----- >>> System check AFTER exe code...")
        lassi.get_power.report_gpu_memory()

    # Convert to native Python float:
    total_energy_joules = float(total_energy_joules)

    return return_result, stdout_output, total_energy_joules, ave_power, std_power, duration, metrics_results_string


# NEW: Add AMD-specific execution function
def execute_code_amd(experiment_data_directory, code_version, zero_shot, trial_number, 
                    directory, executable_file, exe_input_parameters, power_measure, 
                    code_source, target_lang, build_script, make_script, wait=10):
    """
    AMD-specific version of execute_code - mirrors NVIDIA logic but uses AMD power monitoring
    """
    
    if power_measure:
        print("\n----- >>> AMD GPU system check BEFORE exe code...")
        amd_report_gpu_memory()

    duration = 0
    start_power_time = 0
    total_energy_joules = 0
    ave_power = 0
    std_power = 0
    metrics_results_string = ""

    # Simulating a progress bar while sleeping
    for _ in tqdm(range(wait), desc="Waiting for the system to settle down for " + str(wait) + " seconds..."):
        time.sleep(1)

    return_code = ""

    print("executable_file: ", executable_file)
    
    # ... (copy the directory handling logic from original execute_code) ...
    if make_script:
        if directory in str(os.getcwd()):
            initial_dir = os.getcwd()
        else:
            initial_dir = os.path.join(os.getcwd(), directory)
    else:
        initial_dir = os.getcwd()
    print("initial_dir: ", initial_dir)
    
    directory_path = os.path.dirname(executable_file)
    print("directory_path: ", directory_path)
    executable = os.path.basename(executable_file)
    print("executable: ", executable)

    if make_script:
        os.chdir(initial_dir)
        # executable = directory_path + "/" + executable
    else:
        os.chdir(directory_path)
        
    subprocess.run(['chmod', '+x', executable])

    if build_script:
        execute_command = ['./' + executable] + exe_input_parameters[0].split(' ')
    elif make_script:
        print("exe_input_parameters: ", exe_input_parameters)
        os.chdir(initial_dir)
        execute_command = ['./' + executable] + exe_input_parameters
    else:
        execute_command = ['./' + executable] + exe_input_parameters
    print("Working directory: " + os.getcwd())
    print("Execute command: " + str(execute_command) + "\n")

    # AMD Power monitoring setup
    if power_measure:
        amd_monitor = AMDPowerMonitor()
        if amd_monitor.gpu_count == 0:
            print("❌ No AMD GPUs detected for power monitoring")
            power_measure = False
        else:
            # Collect idle power average value before exe
            idle_power_before = []
            for _ in range(200):  # 2 seconds @ 0.01s intervals
                p = amd_monitor.get_power_usage(0)
                idle_power_before.append(p)
                time.sleep(0.01)
            ave_idle_before = sum(idle_power_before) / len(idle_power_before)
     
    # ... (copy the execution logic from original execute_code, but use AMD power monitoring) ...
    
    if os.path.exists(executable):
        print("running: executable - ", executable)
        # if make_script:
            # executable = "README.md"
        try:
            with open(executable, 'rb') as code_file:
                if power_measure:
                    start_power_time = time.time()
                    power_queue = queue.Queue()
                    stop_power_event = threading.Event()
    
                    # Start AMD power collection thread
                    power_thread = Thread(target=amd_power_collection_thread, 
                                        args=(power_queue, stop_power_event, 0))
                    power_thread.start()
                    time.sleep(15)
    
                start_exe_time = time.time() - start_power_time
                repeat = 1
                stdout_lines = []
                stderr_lines = []
                
                for i in range(0, repeat):
                    process = subprocess.Popen(execute_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    exception_error = ""
                    
                    print("\n----- < BEGIN: STANDARD OUTPUT FROM CODE EXECUTION > -----\n")
                    try:
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            print(line, end='')
                            stdout_lines.append(line)
                        process.stdout.close()
                
                        while True:
                            line = process.stderr.readline()
                            if not line:
                                break
                            stderr_lines.append(line)
                        process.stderr.close()
                    
                        return_code = process.wait()
                        
                        if return_code != 0:
                            error_output = ''.join(stderr_lines)
                            print("Program Execution Error:", error_output)
    
                    except Exception as e:
                        print("An error occurred while executing the program: ", e)
                        exception_error = str(e)

                    print("\n----- < END: Standard Output > -----\n")

                total_energy_joules = 0
                ave_power = 0
                
                if power_measure and (return_code == 0 or any("RESULTS" in s for s in stdout_lines)):            
                    end_exe_time = time.time() - start_power_time
                    duration = end_exe_time - start_exe_time
                    time.sleep(15)
                    end_power_time = time.time()            
                    stop_power_event.set()
                    power_thread.join()
                    
                    # Collect idle power average value after exe
                    idle_power_after = []
                    for _ in range(200):
                        p = amd_monitor.get_power_usage(0)
                        idle_power_after.append(p)
                        time.sleep(0.01)
                    ave_idle_after = sum(idle_power_after) / len(idle_power_after)

                    ave_idle_power = (ave_idle_before + ave_idle_after) / 2
        
                    total_energy_joules, ave_power, std_power, metrics_results_string = lassi.get_power.power_collect_data(
                        experiment_data_directory,
                        code_version,
                        zero_shot,
                        trial_number,
                        power_queue, 
                        code_source, 
                        target_lang, 
                        start_exe_time, 
                        end_exe_time,
                        ave_idle_power,
                        idle=False
                    )
        
        except IOError as e:
            print(f"Pipeline error | An I/O error occurred: {e.strerror}")
        except Exception as e:
            print(f"Pipeline error | An error occurred: {e}")
    else:
        print(f"Pipeline error | The file {executable} does not exist.")
    
    # ... (copy the return logic from original execute_code) ...
    return_result = "0"
    if any("RESULTS" in s for s in stdout_lines) or return_code == 0:
        print("\n" + color.BOLD + "Program executed successfully!" + color.END)
    else:
        return_result = "Program exited with return code: " + str(return_code) + " "
        if return_code == -11:
            return_result += "Segmentation fault detected."
        stderr_output = ''.join(stderr_lines)
        if exception_error != "":
            return_result += "\nException Error: " + exception_error
        if stderr_output:
            return_result += "\nStandard Error: " + stderr_output
        print("\n" + color.BOLD + color.RED + "***** Execution failed." + color.END)
        print("Error message: ")
        print(return_result)
    
    os.chdir(initial_dir)
    stdout_output = ''.join(stdout_lines)

    if power_measure:
        print("\n----- >>> AMD GPU system check AFTER exe code...")
        amd_report_gpu_memory()

    total_energy_joules = float(total_energy_joules)

    return return_result, stdout_output, total_energy_joules, ave_power, std_power, duration, metrics_results_string
