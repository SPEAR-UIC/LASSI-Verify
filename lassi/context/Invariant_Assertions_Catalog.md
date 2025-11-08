
---

# Invariant Assertion Catalog for GPU Code Instrumentation

## Critical Distinction: Type A vs Type B Assertions

### ✅ Type A: Invariant Assertions (REQUIRED - INSERT THESE)
Properties that remain true regardless of parallelization strategy or execution order:
- Input validity constraints
- Mathematical invariants (conservation laws, symmetry, physical laws)
- Output validity and correctness
- Numerical properties that must hold
- **These survive ANY refactoring, optimization, or parallelization change**

### ❌ Type B: Implementation-Specific Assertions (FORBIDDEN - NEVER INSERT THESE)
Properties tied to specific execution patterns:
- Thread execution order assumptions
- Intermediate state checks assuming serial or specific parallel execution
- Timing or performance metrics
- Thread/block/warp-specific checks
- Algorithm implementation details that change with optimization

**YOU MUST INSERT ONLY TYPE A ASSERTIONS.**

---

## GPU-Specific Considerations

### Host vs Device Code
- **Host code assertions:** Place assertions in host functions before kernel launches and after device-to-host transfers
- **Device code assertions:** Use `assert()` in device code sparingly - device assertions have performance cost
- **Preferred approach:** Validate invariants on **host side** after transferring results from device

### Memory Transfer Points (Critical)
Insert assertions at these strategic locations:
1. **Before host-to-device transfer:** Validate input data
2. **After device-to-host transfer:** Validate output data
3. **Before kernel launch:** Validate kernel parameters
4. **After kernel completion:** Validate results

### Example GPU Code Instrumentation Pattern:
```cpp
// LASSI_INSTRUMENTATION - Input validation before GPU transfer
assert(h_input != nullptr && N > 0);
for (int i = 0; i < N; i++) {
    assert(isfinite(h_input[i]));
}

cudaMemcpy(d_input, h_input, N * sizeof(double), cudaMemcpyHostToDevice);

kernel<<<blocks, threads>>>(d_input, d_output, N);
cudaDeviceSynchronize();

cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

// LASSI_INSTRUMENTATION - Output validation after GPU transfer
for (int i = 0; i < N; i++) {
    assert(isfinite(h_output[i]));
}
```

---

## Your Instructions

### Step 1: Analyze the Code
Identify:
- Kernel functions and host functions
- Input/output data structures
- Memory transfer points (cudaMemcpy, hipMemcpy)
- Mathematical operations performed
- Physical domain and expected invariants

### Step 2: Select Minimal Relevant Assertions
- Choose from the catalog below
- Focus on properties that validate correctness
- Prefer host-side assertions over device-side
- Consider computational cost (avoid expensive checks in hot loops)

### Step 3: Insert Assertions at Strategic Points
- **Host function start:** Input validation
- **Before device transfer:** Data validity
- **After device transfer:** Output validation
- **Host function end:** Final correctness checks

### Step 4: Mark All Insertions
```cpp
// LASSI_INSTRUMENTATION - [Brief description of what you're checking]
[assertion code here]
```

### Step 5: Preserve Original Code
- Insert assertions around original code
- Do NOT modify the original algorithm
- Do NOT change kernel implementations
- Only ADD assertion checks

---

## ❌ ANTI-PATTERNS: What NOT to Do

These are examples of **INCORRECT** assertion insertions that violate the invariant-only rule:

### ❌ WRONG: Thread Execution Order Assumptions
```cpp
// BAD - Assumes specific thread execution order
__global__ void bad_kernel(double* data, int N) {
    int tid = threadIdx.x;
    // WRONG: This assumes thread 0 executes before thread 1
    if (tid > 0) {
        assert(data[tid-1] < data[tid]); // IMPLEMENTATION-SPECIFIC
    }
}
```

### ❌ WRONG: Device-Side Assertions for Expensive Checks
```cpp
// BAD - Expensive check in device code
__global__ void bad_kernel(double* matrix, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // WRONG: O(N) check per thread = O(N²) total cost
        for (int j = 0; j < N; j++) {
            assert(isfinite(matrix[i*N + j]));
        }
    }
}
```
**CORRECT:** Validate matrix on host after transfer

### ❌ WRONG: Intermediate Algorithm State
```cpp
// BAD - Checks implementation-specific intermediate state
for (int iter = 0; iter < max_iter; iter++) {
    jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x, d_x_new, N);
    
    // WRONG: Monotonicity assumption won't hold for all parallel strategies
    cudaMemcpy(h_x_new, d_x_new, N*sizeof(double), cudaMemcpyDeviceToHost);
    double new_error = compute_error(h_x_new, N);
    assert(new_error < prev_error); // IMPLEMENTATION-SPECIFIC
    prev_error = new_error;
}
```
**CORRECT:** Only check final convergence after all iterations

### ❌ WRONG: Performance Metrics
```cpp
// BAD - Timing assertions are implementation-specific
auto start = std::chrono::high_resolution_clock::now();
kernel<<<blocks, threads>>>(d_data, N);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// WRONG: Performance will vary with refactoring
assert(duration.count() < 100); // IMPLEMENTATION-SPECIFIC
```

### ❌ WRONG: Shared Memory Assumptions
```cpp
// BAD - Shared memory state during execution
__global__ void bad_kernel(double* data, int N) {
    __shared__ double shared_data[256];
    int tid = threadIdx.x;
    
    shared_data[tid] = data[tid];
    __syncthreads();
    
    // WRONG: Assumes specific shared memory layout
    assert(shared_data[0] == data[0]); // IMPLEMENTATION-SPECIFIC
}
```

### ✅ CORRECT Alternatives for Each Case:

**Case 1 (Order):** Check final sorted property on host
```cpp
// GOOD - Check invariant property after kernel completes
cudaMemcpy(h_data, d_data, N*sizeof(double), cudaMemcpyDeviceToHost);
for (int i = 0; i < N-1; i++) {
    assert(h_data[i] <= h_data[i+1]); // Sorted property is invariant
}
```

**Case 2 (Expensive):** Validate on host
```cpp
// GOOD - Single O(N²) check on host
cudaMemcpy(h_matrix, d_matrix, N*N*sizeof(double), cudaMemcpyDeviceToHost);
for (int i = 0; i < N*N; i++) {
    assert(isfinite(h_matrix[i]));
}
```

**Case 3 (Intermediate):** Check final convergence only
```cpp
// GOOD - Check final result satisfies convergence
cudaMemcpy(h_x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
double residual = compute_residual(h_A, h_b, h_x, N);
assert(residual < tol);
```

---

## Assertion Categories

## Category 1: Input Validity (7 patterns)

### 1.1 Pointer Validity
**Use when:** Functions accept pointer arguments
```c
// LASSI_INSTRUMENTATION - Validate input pointers
assert(ptr != NULL && "Input pointer must not be NULL");
```

### 1.2 Array Size Constraints
**Use when:** Functions accept array dimensions
```c
// LASSI_INSTRUMENTATION - Validate array dimensions
assert(N > 0 && M > 0 && "Array dimensions must be positive");
assert(N * M > 0 && "Array dimensions must not overflow");
```

### 1.3 Value Range Constraints
**Use when:** Inputs have valid ranges (probabilities, angles, indices)
```c
// LASSI_INSTRUMENTATION - Validate input ranges
assert(probability >= 0.0 && probability <= 1.0 && "Probability must be in [0,1]");
assert(index >= 0 && index < array_size && "Index must be within array bounds");
assert(angle >= 0.0 && angle < 2.0 * M_PI && "Angle must be in [0, 2π)");
```

### 1.4 Numerical Validity (Initialization)
**Use when:** Checking input data for NaN/Inf
```c
// LASSI_INSTRUMENTATION - Validate numerical inputs
for (int i = 0; i < N; i++) {
    assert(!isnan(array[i]) && !isinf(array[i]) && "Input contains NaN or Inf");
}
```

### 1.5 Buffer Size Adequacy
**Use when:** Output buffers must be large enough
```c
// LASSI_INSTRUMENTATION - Validate buffer sizes
assert(output_size >= required_size && "Output buffer too small");
```

### 1.6 Configuration Validity
**Use when:** Function behavior depends on configuration parameters
```c
// LASSI_INSTRUMENTATION - Validate configuration
assert(num_threads > 0 && num_threads <= MAX_THREADS && "Invalid thread count");
assert(tolerance > 0.0 && tolerance < 1.0 && "Tolerance must be in (0,1)");
```

### 1.7 Data Structure Integrity
**Use when:** Complex data structures are passed as input
```c
// LASSI_INSTRUMENTATION - Validate structure integrity
assert(matrix->rows == matrix->cols && "Matrix must be square");
assert(graph->num_edges <= graph->num_nodes * (graph->num_nodes - 1) / 2 
       && "Invalid graph structure");
```

---

## Category 2: Conservation Laws (5 patterns)

### 2.1 Mass/Energy Conservation
**Use when:** Physical quantities should be conserved
```c
// LASSI_INSTRUMENTATION - Conservation check
double initial_total = compute_total(array, N);
// ... computation ...
double final_total = compute_total(array, N);
double relative_error = fabs(final_total - initial_total) / fabs(initial_total);
assert(relative_error < 1e-6 && "Mass/energy not conserved");
```

### 2.2 Element Count Conservation
**Use when:** Number of elements should not change (transforms, permutations)
```c
// LASSI_INSTRUMENTATION - Element count conservation
int initial_count = count_elements(input, N);
// ... computation ...
int final_count = count_elements(output, M);
assert(initial_count == final_count && "Element count not preserved");
```

### 2.3 Sum Conservation
**Use when:** Total sum should remain constant
```c
// LASSI_INSTRUMENTATION - Sum conservation
double initial_sum = 0.0;
for (int i = 0; i < N; i++) initial_sum += array[i];
// ... computation ...
double final_sum = 0.0;
for (int i = 0; i < N; i++) final_sum += array[i];
assert(fabs(final_sum - initial_sum) < 1e-6 * fabs(initial_sum) 
       && "Sum not conserved");
```

### 2.4 Norm Conservation
**Use when:** Vector/matrix norms should be preserved (unitary transforms)
```c
// LASSI_INSTRUMENTATION - Norm conservation
double initial_norm = compute_l2_norm(vector, N);
// ... computation ...
double final_norm = compute_l2_norm(vector, N);
assert(fabs(final_norm - initial_norm) < 1e-6 && "L2 norm not preserved");
```

### 2.5 Probability Distribution Conservation
**Use when:** Working with probability distributions
```c
// LASSI_INSTRUMENTATION - Probability conservation
double sum = 0.0;
for (int i = 0; i < N; i++) {
    assert(probs[i] >= 0.0 && "Probabilities must be non-negative");
    sum += probs[i];
}
assert(fabs(sum - 1.0) < 1e-6 && "Probabilities must sum to 1");
```

---

## Category 3: Symmetry Properties (4 patterns)

### 3.1 Matrix Symmetry
**Use when:** Operations should preserve symmetric matrices
```c
// LASSI_INSTRUMENTATION - Matrix symmetry check
for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
        double diff = fabs(matrix[i*N + j] - matrix[j*N + i]);
        assert(diff < 1e-10 && "Matrix symmetry violated");
    }
}
```

### 3.2 Permutation Symmetry
**Use when:** Result should be invariant under permutation of symmetric inputs
```c
// LASSI_INSTRUMENTATION - Permutation invariance
// For commutative operations like addition or multiplication
assert(result_AB == result_BA && "Operation should be commutative");
```

### 3.3 Reflection Symmetry
**Use when:** Physical system has reflection symmetry
```c
// LASSI_INSTRUMENTATION - Reflection symmetry
for (int i = 0; i < N/2; i++) {
    double diff = fabs(field[i] - field[N-1-i]);
    assert(diff < tolerance && "Reflection symmetry violated");
}
```

### 3.4 Rotational Symmetry
**Use when:** System has rotational invariance
```c
// LASSI_INSTRUMENTATION - Rotational invariance
// For 2D field with rotational symmetry
double center_value = field[N/2][N/2];
for (int r = 1; r < N/2; r++) {
    // Check points at same radius have similar values
    // (simplified example - actual implementation depends on coordinate system)
}
```

---

## Category 4: Monotonicity (4 patterns)

### 4.1 Non-Decreasing Sequence
**Use when:** Output should be sorted or monotonically increasing
```c
// LASSI_INSTRUMENTATION - Monotonicity check (non-decreasing)
for (int i = 0; i < N - 1; i++) {
    assert(array[i] <= array[i+1] && "Array should be non-decreasing");
}
```

### 4.2 Non-Increasing Sequence
**Use when:** Values should decrease (e.g., residuals, errors)
```c
// LASSI_INSTRUMENTATION - Monotonicity check (non-increasing)
for (int i = 0; i < N - 1; i++) {
    assert(array[i] >= array[i+1] && "Array should be non-increasing");
}
```

### 4.3 Strictly Decreasing Energy/Error
**Use when:** Iterative algorithms should reduce error or energy
```c
// LASSI_INSTRUMENTATION - Convergence monotonicity
assert(current_error < previous_error * (1.0 + 1e-10) 
       && "Error should decrease monotonically");
```

### 4.4 Bounded Monotonicity
**Use when:** Values increase/decrease but within bounds
```c
// LASSI_INSTRUMENTATION - Bounded monotonic increase
for (int i = 0; i < N - 1; i++) {
    assert(array[i] <= array[i+1] && "Should be non-decreasing");
    assert(array[i+1] - array[i] <= max_increment && "Increment too large");
}
```

---

## Category 5: Physical Constraints (6 patterns)

### 5.1 Positivity
**Use when:** Physical quantities must be positive (temperature, density, energy)
```c
// LASSI_INSTRUMENTATION - Physical positivity constraint
for (int i = 0; i < N; i++) {
    assert(temperature[i] > 0.0 && "Temperature must be positive");
    assert(density[i] > 0.0 && "Density must be positive");
}
```

### 5.2 Non-Negativity
**Use when:** Quantities cannot be negative (counts, concentrations, probabilities)
```c
// LASSI_INSTRUMENTATION - Non-negativity constraint
for (int i = 0; i < N; i++) {
    assert(count[i] >= 0 && "Count cannot be negative");
    assert(concentration[i] >= 0.0 && "Concentration cannot be negative");
}
```

### 5.3 Upper Bounds
**Use when:** Physical quantities have maximum values
```c
// LASSI_INSTRUMENTATION - Upper bound constraint
for (int i = 0; i < N; i++) {
    assert(occupancy[i] <= 1.0 && "Occupancy cannot exceed 1");
    assert(efficiency[i] <= 1.0 && "Efficiency cannot exceed 100%");
}
```

### 5.4 Speed of Light Constraint
**Use when:** Velocities in relativistic simulations
```c
// LASSI_INSTRUMENTATION - Relativistic constraint
const double C = 299792458.0; // Speed of light in m/s
for (int i = 0; i < N; i++) {
    double speed = sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
    assert(speed <= C && "Velocity cannot exceed speed of light");
}
```

### 5.5 Thermodynamic Constraints
**Use when:** Working with thermodynamic quantities
```c
// LASSI_INSTRUMENTATION - Thermodynamic constraints
assert(temperature > 0.0 && "Temperature must be positive (Kelvin scale)");
assert(entropy_change >= 0.0 && "Entropy cannot decrease in isolated system");
assert(pressure > 0.0 && "Pressure must be positive");
```

### 5.6 Charge Conservation
**Use when:** Electromagnetic simulations
```c
// LASSI_INSTRUMENTATION - Charge conservation
double initial_charge = compute_total_charge(particles, N);
// ... computation ...
double final_charge = compute_total_charge(particles, N);
assert(fabs(final_charge - initial_charge) < 1e-10 && "Charge not conserved");
```

---

## Category 6: Numerical Stability (5 patterns)

### 6.1 NaN Detection (Output)
**Use when:** Validating output for numerical errors
```c
// LASSI_INSTRUMENTATION - NaN detection
for (int i = 0; i < N; i++) {
    assert(!isnan(result[i]) && "Result contains NaN");
}
```

### 6.2 Infinity Detection
**Use when:** Checking for overflow/divergence
```c
// LASSI_INSTRUMENTATION - Infinity detection
for (int i = 0; i < N; i++) {
    assert(!isinf(result[i]) && "Result contains Infinity");
}
```

### 6.3 Finite Value Check (Combined)
**Use when:** Ensuring all output values are finite
```c
// LASSI_INSTRUMENTATION - Finite value check
for (int i = 0; i < N; i++) {
    assert(isfinite(result[i]) && "Result must be finite");
}
```

### 6.4 Condition Number Bounds
**Use when:** Matrix operations sensitive to conditioning
```c
// LASSI_INSTRUMENTATION - Conditioning check
double condition_number = estimate_condition_number(matrix, N);
assert(condition_number < 1e12 && "Matrix is ill-conditioned");
```

### 6.5 Relative Error Bounds
**Use when:** Comparing computed result to known or expected value
```c
// LASSI_INSTRUMENTATION - Relative error bound
double relative_error = fabs(computed - expected) / fabs(expected);
assert(relative_error < tolerance && "Relative error exceeds tolerance");
```

---

## Category 7: Structural Invariants (4 patterns)

### 7.1 Sparsity Pattern Preservation
**Use when:** Sparse matrix operations should preserve zero structure
```c
// LASSI_INSTRUMENTATION - Sparsity preservation
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        if (original_matrix[i*N + j] == 0.0) {
            assert(fabs(result_matrix[i*N + j]) < 1e-15 
                   && "Sparsity pattern violated");
        }
    }
}
```

### 7.2 Diagonal Dominance
**Use when:** Certain numerical methods require diagonally dominant matrices
```c
// LASSI_INSTRUMENTATION - Diagonal dominance check
for (int i = 0; i < N; i++) {
    double diag = fabs(matrix[i*N + i]);
    double row_sum = 0.0;
    for (int j = 0; j < N; j++) {
        if (i != j) row_sum += fabs(matrix[i*N + j]);
    }
    assert(diag >= row_sum && "Matrix not diagonally dominant");
}
```

### 7.3 Positive Definiteness
**Use when:** Matrix should remain positive definite (covariance, Gram matrices)
```c
// LASSI_INSTRUMENTATION - Positive definiteness check
// Check all diagonal elements are positive (necessary condition)
for (int i = 0; i < N; i++) {
    assert(matrix[i*N + i] > 0.0 && "Diagonal must be positive");
}
// For full check, would need eigenvalue computation (expensive)
```

### 7.4 Orthogonality
**Use when:** Vectors or matrices should be orthogonal
```c
// LASSI_INSTRUMENTATION - Orthogonality check
double dot_product = 0.0;
for (int i = 0; i < N; i++) {
    dot_product += vector1[i] * vector2[i];
}
assert(fabs(dot_product) < 1e-10 && "Vectors should be orthogonal");
```

---

## Selection Guidelines for GPU Code

### DO Insert Assertions For:
- ✅ **Host-side input validation** (before cudaMemcpy H→D)
- ✅ **Host-side output validation** (after cudaMemcpy D→H)
- ✅ **Kernel launch parameter validation** (grid/block dimensions)
- ✅ **Mathematical invariants** (conservation, symmetry, physical laws)
- ✅ **Properties that survive ANY parallelization or optimization**

### DO NOT Insert Assertions For:
- ❌ Thread/block/warp execution order
- ❌ Shared memory state during kernel execution
- ❌ Thread synchronization points
- ❌ Performance metrics (kernel timing, memory bandwidth)
- ❌ Implementation-specific GPU optimizations
- ❌ Intermediate kernel states (unless copied to host)

### GPU-Specific Guidelines:

**Assertion Placement Priority:**
1. **Host-side (preferred):** All invariant checks on CPU before/after GPU operations
2. **Device-side (use sparingly):** Only for critical invariants that must be checked per-thread
   - Device assertions have significant performance cost
   - May cause kernel to fail silently or hang
   - Prefer copying data to host for validation

**Memory Transfer Validation:**
```cpp
// ALWAYS validate before H→D transfer
assert(h_data != nullptr);
for (int i = 0; i < N; i++) assert(isfinite(h_data[i]));
cudaMemcpy(d_data, h_data, N*sizeof(double), cudaMemcpyHostToDevice);

// ALWAYS validate after D→H transfer  
cudaMemcpy(h_result, d_result, N*sizeof(double), cudaMemcpyDeviceToHost);
for (int i = 0; i < N; i++) assert(isfinite(h_result[i]));
```

**Kernel Parameter Validation:**
```cpp
// LASSI_INSTRUMENTATION - Validate kernel parameters
assert(threads > 0 && threads <= 1024 && "Invalid thread count");
assert(blocks > 0 && "Invalid block count");
assert(N == blocks * threads || N < blocks * threads && "Grid size mismatch");
```

### Cost Considerations:
- **Low cost:** Pointer checks, single value checks, kernel parameter validation
- **Medium cost:** O(N) checks on host after device-to-host transfer
- **High cost:** O(N²) matrix checks on host
- **Very high cost:** Device-side assertions (avoid if possible)

**For large-scale GPU codes:**
- Prefer low-medium cost host-side assertions
- Avoid expensive checks in performance-critical paths
- Consider sampling for very large arrays (check subset of elements)
- Use compile-time flags to disable assertions in production

---

## Domain-Specific Guidance

### Computational Fluid Dynamics (CFD)
Key invariants:
- Mass conservation
- Energy conservation
- Momentum conservation
- Positivity of density, pressure, temperature
- Speed limits (subsonic/supersonic bounds)

### Linear Algebra
Key invariants:
- Matrix symmetry preservation
- Orthogonality preservation
- Norm conservation (unitary transforms)
- Rank preservation (where applicable)
- Conditioning bounds

### Molecular Dynamics
Key invariants:
- Particle count conservation
- Energy conservation (microcanonical ensemble)
- Momentum conservation
- Positivity of inter-particle distances
- Temperature bounds

### Monte Carlo Simulations
Key invariants:
- Probability sum to 1
- Non-negativity of probabilities
- Detailed balance (where applicable)
- Acceptance ratio bounds

### Iterative Solvers
Key invariants:
- Residual decrease (monotonicity)
- Solution bounds
- Convergence criteria satisfaction
- Numerical stability

---

## Example: GPU Jacobi Solver Instrumentation

**INPUT CODE (before instrumentation):**
```cpp
__global__ void jacobi_kernel(double* A, double* b, double* x, double* x_new, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) sigma += A[i*N + j] * x[j];
        }
        x_new[i] = (b[i] - sigma) / A[i*N + i];
    }
}

void jacobi_solver(double* A, double* b, double* x, int N, int max_iter, double tol) {
    double *d_A, *d_b, *d_x, *d_x_new;
    
    cudaMalloc(&d_A, N*N*sizeof(double));
    cudaMalloc(&d_b, N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_x_new, N*sizeof(double));
    
    cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    for (int iter = 0; iter < max_iter; iter++) {
        jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x, d_x_new, N);
        cudaDeviceSynchronize();
        std::swap(d_x, d_x_new);
    }
    
    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_x_new);
}
```

**YOUR OUTPUT (after instrumentation):**
```cpp
__global__ void jacobi_kernel(double* A, double* b, double* x, double* x_new, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) sigma += A[i*N + j] * x[j];
        }
        x_new[i] = (b[i] - sigma) / A[i*N + i];
    }
}

void jacobi_solver(double* A, double* b, double* x, int N, int max_iter, double tol) {
    // LASSI_INSTRUMENTATION - Input validation (host-side)
    assert(A != nullptr && b != nullptr && x != nullptr && "Input pointers must be valid");
    assert(N > 0 && "Matrix dimension must be positive");
    assert(max_iter > 0 && "Max iterations must be positive");
    assert(tol > 0.0 && tol < 1.0 && "Tolerance must be in (0,1)");
    
    // LASSI_INSTRUMENTATION - Input numerical validity
    for (int i = 0; i < N*N; i++) {
        assert(isfinite(A[i]) && "Matrix A contains non-finite values");
    }
    for (int i = 0; i < N; i++) {
        assert(isfinite(b[i]) && "Vector b contains non-finite values");
        assert(isfinite(x[i]) && "Initial guess x contains non-finite values");
    }
    
    // LASSI_INSTRUMENTATION - Diagonal dominance check (Jacobi requirement)
    for (int i = 0; i < N; i++) {
        assert(fabs(A[i*N + i]) > 1e-14 && "Diagonal element too small");
    }
    
    // LASSI_INSTRUMENTATION - Matrix symmetry (if applicable to problem)
    bool check_symmetry = true; // Set based on problem characteristics
    if (check_symmetry) {
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                assert(fabs(A[i*N + j] - A[j*N + i]) < 1e-10 && "Matrix should be symmetric");
            }
        }
    }
    
    double *d_A, *d_b, *d_x, *d_x_new;
    
    cudaMalloc(&d_A, N*N*sizeof(double));
    cudaMalloc(&d_b, N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_x_new, N*sizeof(double));
    
    cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    for (int iter = 0; iter < max_iter; iter++) {
        jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x, d_x_new, N);
        cudaDeviceSynchronize();
        std::swap(d_x, d_x_new);
    }
    
    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
    
    // LASSI_INSTRUMENTATION - Output validation (host-side after device-to-host transfer)
    for (int i = 0; i < N; i++) {
        assert(isfinite(x[i]) && "Solution x contains non-finite values");
    }
    
    // LASSI_INSTRUMENTATION - Residual check (validates solution quality)
    double residual_norm = 0.0;
    for (int i = 0; i < N; i++) {
        double ax_i = 0.0;
        for (int j = 0; j < N; j++) {
            ax_i += A[i*N + j] * x[j];
        }
        double r_i = b[i] - ax_i;
        residual_norm += r_i * r_i;
    }
    residual_norm = sqrt(residual_norm);
    assert(residual_norm < tol * 10.0 && "Solution does not satisfy convergence criterion");
    
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_x_new);
}
```

**Key Points Demonstrated:**
1. ✅ Assertions placed on **host side** (not in kernel)
2. ✅ Validation **before** host-to-device transfer
3. ✅ Validation **after** device-to-host transfer
4. ✅ All assertions use standard library (`assert.h`, `math.h`)
5. ✅ Assertions check **invariant properties** only
6. ✅ Original kernel code **unchanged**
7. ✅ Clear `// LASSI_INSTRUMENTATION` markers

---

## Notes on Implementation

1. **All assertions use standard library functions only:**
   - `<assert.h>` for assert()
   - `<math.h>` for isnan(), isinf(), isfinite(), fabs(), sqrt()
   - No custom LASSI library required

2. **Assertions marked with `// LASSI_INSTRUMENTATION`:**
   - This marker tells the code generation LLM to preserve these sections
   - Enables automated verification that assertions weren't modified

3. **Rationale documentation:**
   - Each assertion block should include a brief comment explaining what invariant is being checked
   - This aids in debugging if assertions fail

4. **Tolerance values:**
   - Use appropriate tolerances for floating-point comparisons
   - Typical values: 1e-6 to 1e-10 depending on numerical precision requirements
   - Relative tolerances preferred for conservation laws

---
