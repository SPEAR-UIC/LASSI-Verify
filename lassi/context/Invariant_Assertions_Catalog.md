# Invariant Assertion Catalog

## Core Principle
Insert **ONLY invariant assertions** (properties that survive ANY refactoring/parallelization). Never insert implementation-specific checks.

## ❌ FORBIDDEN Patterns
```cpp
// NEVER wrap in conditionals
#ifdef DEBUG
    assert(...);  // WRONG
#endif

// NEVER check implementation details
assert(block.x == 16);      // WRONG - specific block size
assert(threads == 256);     // WRONG - specific thread count
assert(grid.x == N/16);     // WRONG - specific grid config

// NEVER check execution order
if (tid > 0) assert(data[tid-1] < data[tid]);  // WRONG - thread order

// NEVER expensive device checks
__global__ void kernel() {
    for (int j = 0; j < N; j++) assert(...);  // WRONG - O(N) per thread
}
```

## ✅ Required Patterns

### 1. Input Validation (Host-Side, Before H→D Transfer)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
// LASSI_INSTRUMENTATION - Input validation
assert(ptr != nullptr && N > 0 && M > 0);

// Small arrays: full check
for (int i = 0; i < N; i++) {
    assert(isfinite(array[i]));
}

// Large arrays (≥1M elements): sample first/last 1000
for (int i = 0; i < 1000 && i < N; i++) {
    assert(isfinite(array[i]));
}
for (int i = (N > 1000 ? N - 1000 : 0); i < N; i++) {
    assert(isfinite(array[i]));
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 2. Kernel Parameters (General Constraints Only)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
// LASSI_INSTRUMENTATION - Kernel parameters
assert(block.x > 0 && block.y > 0 && block.x * block.y <= 1024);
assert(grid.x > 0 && grid.y > 0);
assert(grid.x * block.x >= N && grid.y * block.y >= N);
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 3. Output Validation (Host-Side, After D→H Transfer)
```cpp
cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
// LASSI_INSTRUMENTATION - Output validation
// Finite values (use sampling for large arrays)
for (int i = 0; i < N; i++) {
    assert(isfinite(h_output[i]));
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

## Domain-Specific Invariants

**Select patterns based on code analysis:**

### Conservation Laws
```cpp
// Mass/energy/sum conservation
double initial = compute_sum(array, N);
// ... computation ...
double final = compute_sum(array, N);
assert(fabs(final - initial) / fabs(initial) < 1e-6);
```

### Symmetry
```cpp
// Matrix symmetry
for (int i = 0; i < N; i++) {
    for (int j = i + 1; j < N; j++) {
        assert(fabs(matrix[i*N+j] - matrix[j*N+i]) < 1e-10);
    }
}
```

### Physical Constraints
```cpp
// Positivity (temperature, density, energy)
for (int i = 0; i < N; i++) {
    assert(temperature[i] > 0.0 && density[i] > 0.0);
}

// Non-negativity (counts, probabilities)
assert(count >= 0 && probability >= 0.0 && probability <= 1.0);

// Boundary conditions preserved
assert(fabs(field[boundary_idx] - expected_value) < tolerance);
```

### Monotonicity
```cpp
// Sorted/non-decreasing
for (int i = 0; i < N - 1; i++) {
    assert(array[i] <= array[i+1]);
}

// Error decrease (final check only, not per-iteration)
assert(final_error < tolerance);
```

### Numerical Stability
```cpp
// Finite values (no NaN/Inf)
assert(isfinite(value) && !isnan(value) && !isinf(value));

// Condition number bounds
assert(condition_number < 1e12);
```

### Structural Invariants
```cpp
// Diagonal dominance (if required by method)
for (int i = 0; i < N; i++) {
    double diag = fabs(matrix[i*N + i]);
    double row_sum = 0.0;
    for (int j = 0; j < N; j++) {
        if (i != j) row_sum += fabs(matrix[i*N + j]);
    }
    assert(diag >= row_sum);
}

// Diagonal positivity (for positive definite)
for (int i = 0; i < N; i++) {
    assert(matrix[i*N + i] > 0.0);
}
```

## Selection Strategy

1. **Analyze code domain** (CFD, linear algebra, iterative solver, etc.)
2. **Select 3-5 relevant patterns** from above
3. **Place at strategic points:**
   - After input initialization → input validation
   - After kernel parameters setup → parameter validation
   - After final D→H transfer → output validation
4. **Use sampling for arrays > 1M elements**
5. **Always use protective comments**

## Domain Quick Reference

- **CFD:** conservation (mass/energy/momentum), positivity (density/pressure/temp)
- **Linear Algebra:** symmetry, orthogonality, norm conservation, conditioning
- **Iterative Solvers:** convergence, residual bounds, monotonicity (final only)
- **Molecular Dynamics:** particle count, energy conservation, positivity (distances)
- **Monte Carlo:** probability sum to 1, non-negativity

## Example: Jacobi Solver

```cpp
void jacobi_solver(float* A, float* b, float* x, int N, int max_iter, float tol) {
    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
    // LASSI_INSTRUMENTATION - Input validation
    assert(A != nullptr && b != nullptr && x != nullptr && N > 0);
    for (int i = 0; i < N*N; i++) assert(isfinite(A[i]));
    for (int i = 0; i < N; i++) assert(isfinite(b[i]) && isfinite(x[i]));
    // Diagonal non-zero (Jacobi requirement)
    for (int i = 0; i < N; i++) assert(fabs(A[i*N + i]) > 1e-14);
    // <<< END LASSI-VERIFY ASSERTIONS <<<
    
    // ... allocate device memory ...
    // ... copy to device ...
    
    dim3 grid(N/16, N/16);
    dim3 block(16, 16);
    
    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
    // LASSI_INSTRUMENTATION - Kernel parameters
    assert(block.x > 0 && block.y > 0 && block.x * block.y <= 1024);
    assert(grid.x * block.x >= N && grid.y * block.y >= N);
    // <<< END LASSI-VERIFY ASSERTIONS <<<
    
    // ... iteration loop ...
    
    cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
    // LASSI_INSTRUMENTATION - Output validation
    for (int i = 0; i < N; i++) assert(isfinite(x[i]));
    // Residual check
    float residual = 0.0;
    for (int i = 0; i < N; i++) {
        float ax = 0.0;
        for (int j = 0; j < N; j++) ax += A[i*N + j] * x[j];
        float r = b[i] - ax;
        residual += r * r;
    }
    residual = sqrtf(residual / N);
    assert(residual < tol * 10.0);
    // <<< END LASSI-VERIFY ASSERTIONS <<<
    
    // ... cleanup ...
}
```

## Critical Reminders

1. Host-side only (never in kernels)
2. No conditional compilation (#ifdef DEBUG)
3. No implementation-specific checks (block.x == 16)
4. Use sampling for large arrays (>1M elements)
5. Always use protective comments
6. Standard library only (assert.h, cmath)