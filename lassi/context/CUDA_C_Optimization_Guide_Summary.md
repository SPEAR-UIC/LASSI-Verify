**Note: High Priority:** Minimize data transfer between the host and the device, even if it means running some kernels on the device that do not show performance gains when compared with running them on the host CPU.

Intermediate data structures should be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory.

Also, because of the overhead associated with each transfer, batching many small transfers into one larger transfer performs significantly better than making each transfer separately, even if doing so requires packing non-contiguous regions of memory into a contiguous buffer and then unpacking after the transfer.

Finally, higher bandwidth between the host and the device is achieved when using *page-locked* (or *pinned*) memory, as discussed in the CUDA C++ Programming Guide and the Pinned Memory section of this document.

**Note: Low Priority:** Use zero-copy operations on integrated GPUs for CUDA Toolkit version 2.2 and later.

The host code in Zero-copy host code shows how zero copy is typically set up.

**Note: Medium Priority:** To hide latency arising from register dependencies, maintain sufficient numbers of active threads per multiprocessor (i.e., sufficient occupancy).

**Note: Medium Priority:** The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing.

**Note: Low Priority:** Use shift operations to avoid expensive division and modulo calculations.

Integer division and modulo operations are particularly costly and should be avoided or replaced with bitwise operations whenever possible: If *n* is a power of 2, ( *i*/*n* ) is equivalent to ( *i ≫ log*2(*n*) ) and ( *i*%*n* ) is equivalent to ( *i*&(*n −* 1) ).

The compiler will perform these conversions if n is literal. (For further information, refer to Performance Guidelines in the CUDA C++ Programming Guide).

**Note: Low Medium Priority:** Use signed integers rather than unsigned integers as loop counters.

In the C language standard, unsigned integer overflow semantics are well defined, whereas signed integer overflow causes undefined results. Therefore, the compiler can optimize more aggressively with signed arithmetic than it can with unsigned arithmetic. This is of particular note with loop counters: since it is common for loop counters to have values that are always positive, it may be tempting to declare the counters as unsigned. For slightly better performance, however, they should instead be declared as signed.

For example, consider the following code:
    for (i = 0; i < n; i++) {
        out[i] = in[offset + stride*i];
    }

Here, the sub-expression stride*i could overflow a 32-bit integer, so if i is declared as unsigned, the overflow semantics prevent the compiler from using some optimizations that might otherwise have applied, such as strength reduction. If instead i is declared as signed, where the overflow semantics are undefined, the compiler has more leeway to use these optimizations.

**Note: Low Priority:** Avoid automatic conversion of doubles to floats.

The compiler must on occasion insert conversion instructions, introducing additional execution cycles. This is the case for:

* ▶ Functions operating on char or short whose operands generally need to be converted to an int
* ▶ Double-precision floating-point constants (defined without any type suffix) used as input to single-precision floating-point computations

The latter case can be avoided by using single-precision floating-point constants, defined with an f suffix such as 3.141592653589793f, 1.0f, 0.5f.

For single-precision code, use of the float type and the single-precision math functions are highly recommended.

It should also be noted that the CUDA math library's complementary error function, erfcf(), is particularly fast with full single-precision accuracy.

**Note: Medium Priority:** Use the fast math library whenever speed trumps precision.

Two types of runtime math operations are supported. They can be distinguished by their names: some have names with prepended underscores, whereas others do not (e.g., __functionName() versus functionName()). Functions following the __functionName() naming convention map directly to the hardware level. They are faster but provide somewhat lower accuracy (e.g., __sinf(x) and __expf(x)). Functions following functionName() naming convention are slower but have higher accuracy (e.g., sinf(x) and expf(x)). The throughput of __sinf(x), __cosf(x), and__expf(x) is much greater than that of sinf(x), cosf(x), and expf(x). The latter become even more expensive (about an order of magnitude slower) if the magnitude of the argument x needs to be reduced. Moreover, in such cases, the argument-reduction code uses local memory, which can affect performance even more because of the high latency of local memory. More details are available in the *CUDA C++ Programming Guide*.

Note also that whenever sine and cosine of the same argument are computed, the sincos family of instructions should be used to optimize performance:

* ▶ __sincosf() for single-precision fast math (see next paragraph)
* ▶ sincosf() for regular single-precision
* ▶ sincos() for double precision

The -use_fast_math compiler option of nvcc coerces every functionName() call to the equivalent __functionName() call. It also disables single-precision denormal support and lowers the precision of single-precision division in general. This is an aggressive optimization that can both reduce numerical accuracy and alter special case handling. A more robust approach is to selectively introduce calls to fast intrinsic functions only if merited by performance gains and where altered behavior can be tolerated. Note this switch is effective only on single-precision floating point.

**Note: Medium Priority:** Prefer faster, more specialized math functions over slower, more general ones when possible.

For small integer powers (e.g., *x2* or *x3*), explicit multiplication is almost certainly faster than the use of general exponentiation routines such as pow(). While compiler optimization improvements continually seek to narrow this gap, explicit multiplication (or the use of an equivalent purpose-built inline function or macro) can have a significant advantage. This advantage is increased when several powers of the same base are needed (e.g., where both *x2* and *x5* are calculated in close proximity), as this aids the compiler in its common sub-expression elimination (CSE) optimization.

For exponentiation using base 2 or 10, use the functions exp2() or expf2() and exp10() or expf10() rather than the functions pow() or powf(). Both pow() and powf() are heavy-weight functions in terms of register pressure and instruction count due to the numerous special cases arising in general exponentiation and the difficulty of achieving good accuracy across the entire ranges of the base and the exponent. The functions exp2(), exp2f(), exp10(), and exp10f(), on the other hand, are similar to exp() and expf() in terms of performance, and can be as much as ten times faster than their pow()/powf() equivalents.

For exponentiation with an exponent of 1/3, use the cbrt() or cbrtf() function rather than the generic exponentiation functions pow() or powf(), as the former are significantly faster than the latter. Likewise, for exponentation with an exponent of -1/3, use rcbrt() or rcbrtf().

Replace sin(π*<expr>) with sinpi(<expr>), cos(π*<expr>) with cospi(<expr>), and sincos(π*<expr>) with sincospi(<expr>). This is advantageous with regard to both accuracy and performance. As a particular example, to evaluate the sine function in degrees instead of radians, use sinpi(x∕180.0). Similarly, the single-precision functions sinpif(), cospif(), and sincospif() should replace calls to sinf(), cosf(), and sincosf() when the function argument is of the form π*<expr>. (The performance advantage sinpi() has over sin() is due to simplified argument reduction; the accuracy advantage is because sinpi() multiplies by π only implicitly, effectively using an infinitely precise mathematical π rather than a single- or double-precision approximation thereof.)

**Note: High Priority:** Minimize the use of global memory. Prefer shared memory access where possible.

Memory instructions include any instruction that reads from or writes to shared, local, or global memory. When accessing uncached local or global memory, there are hundreds of clock cycles of memory latency.

As an example, the assignment operator in the following sample code has a high throughput, but, crucially, there is a latency of hundreds of clock cycles to read data from global memory:
    __shared__ float shared[32];
    __device__ float device[32];
    shared[threadIdx.x] = device[threadIdx.x];

Much of this global memory latency can be hidden by the thread scheduler if there are sufficient independent arithmetic instructions that can be issued while waiting for the global memory access to complete. However, it is best to avoid accessing global memory whenever possible.

**Note: High Priority:** Avoid different execution paths within the same warp.

Flow control instructions (if, switch, do, for, while) can significantly affect the instruction throughput by causing threads of the same warp to diverge; that is, to follow different execution paths. If this happens, the different execution paths must be executed separately; this increases the total number of instructions executed for this warp.

To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps.

This is possible because the distribution of the warps across the block is deterministic as mentioned in SIMT Architecture of the CUDA C++ Programming Guide. A trivial example is when the controlling condition depends only on (threadIdx / WSIZE) where WSIZE is the warp size.

In this case, no warp diverges because the controlling condition is perfectly aligned with the warps.

For branches including just a few instructions, warp divergence generally results in marginal performance losses. For example, the compiler may use predication to avoid an actual branch. Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads execute the instructions. Threads with a false predicate do not write results, and also do not evaluate addresses or read operands.

Starting with the Volta architecture, Independent Thread Scheduling allows a warp to remain diverged outside of the data-dependent conditional block. An explicit __syncwarp() can be used to guarantee that the warp has reconverged for subsequent instructions.

**Note: Low Priority:** Make it easy for the compiler to use branch predication in lieu of loops or control statements.

Sometimes, the compiler may unroll loops or optimize out if or switch statements by using branch predication instead. In these cases, no warp can ever diverge. The programmer can also control loop unrolling using

#### *#pragma unroll*

For more information on this pragma, refer to the CUDA C++ Programming Guide.

When using branch predication, none of the instructions whose execution depends on the controlling condition is skipped. Instead, each such instruction is associated with a per-thread condition code or predicate that is set to true or false according to the controlling condition. Although each of these instructions is scheduled for execution, only the instructions with a true predicate are actually executed. Instructions with a false predicate do not write results, and they also do not evaluate addresses or read operands.

The compiler replaces a branch instruction with predicated instructions only if the number of instructions controlled by the branch condition is less than or equal to a certain threshold.
