
/** Refactored GPU solver for 2D lid-driven cavity problem, focusing on energy efficiency
 * \file main_gpu_refactored.cu
 *
 * This refactored code applies several optimizations to improve energy efficiency,
 * including minimizing host-device data transfers, performing reductions on the GPU,
 * using pinned memory for faster data transfers, optimizing memory accesses,
 * avoiding warp divergence, and precomputing constants.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// Problem size along one side; total number of cells is this squared
#define NUM 512
#define NUM_2 (NUM >> 1)  // Added definition for NUM_2

// Block size, adjusted to 256 for better occupancy
#define BLOCK_SIZE 256

// Double precision
#define DOUBLE

#ifdef DOUBLE
#define Real double

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

#define SMALL 1.0e-10;

/** Reynolds number */
const Real Re_num = 1000.0;

/** SOR relaxation parameter */
const Real omega = 1.7;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9;

/** Safety factor for time step modification */
const Real tau = 0.5;

/** Body forces in x- and y- directions */
const Real gx = 0.0;
const Real gy = 0.0;

/** Domain size (non-dimensional) */
#define xLength 1.0
#define yLength 1.0
#else
#define Real float

// Replace double functions with float versions
#undef fmin
#define fmin fminf
#undef fmax
#define fmax fmaxf
#undef fabs
#define fabs fabsf
#undef sqrt
#define sqrt sqrtf

#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SMALL 1.0e-10f;

/** Reynolds number */
const Real Re_num = 1000.0f;

/** SOR relaxation parameter */
const Real omega = 1.7f;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9f;

/** Safety factor for time step modification */
const Real tau = 0.5f;

/** Body forces in x- and y- directions */
const Real gx = 0.0f;
const Real gy = 0.0f;

/** Domain size (non-dimensional) */
#define xLength 1.0f
#define yLength 1.0f
#endif

/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

/** Macros for easy array access */
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

// Function to set boundary conditions on the host
__host__ void set_BCs_host(Real* u, Real* v)
{
    int ind;

    // Loop through rows and columns
    for (ind = 0; ind < NUM + 2; ++ind) {
        // Left boundary
        u(0, ind) = ZERO;
        v(0, ind) = -v(1, ind);

        // Right boundary
        u(NUM, ind) = ZERO;
        v(NUM + 1, ind) = -v(NUM, ind);

        // Bottom boundary
        u(ind, 0) = -u(ind, 1);
        v(ind, 0) = ZERO;

        // Top boundary
        u(ind, NUM + 1) = TWO - u(ind, NUM);
        v(ind, NUM) = ZERO;

        if (ind == NUM) {
            // Corners
            // Left boundary
            u(0, 0) = ZERO;
            v(0, 0) = -v(1, 0);
            u(0, NUM + 1) = ZERO;
            v(0, NUM + 1) = -v(1, NUM + 1);

            // Right boundary
            u(NUM, 0) = ZERO;
            v(NUM + 1, 0) = -v(NUM, 0);
            u(NUM, NUM + 1) = ZERO;
            v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

            // Bottom boundary
            u(0, 0) = -u(0, 1);
            v(0, 0) = ZERO;
            u(NUM + 1, 0) = -u(NUM + 1, 1);
            v(NUM + 1, 0) = ZERO;

            // Top boundary
            u(0, NUM + 1) = TWO - u(0, NUM);
            v(0, NUM) = ZERO;
            u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
            v(ind, NUM + 1) = ZERO;
        } // end if
    }     // end for
} // end set_BCs_host

// Function to set boundary conditions on the device
__global__ void set_BCs(Real* __restrict__ u, Real* __restrict__ v)
{
    int ind = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

    if (ind <= NUM) {
        // Left boundary
        u(0, ind) = ZERO;
        v(0, ind) = -v(1, ind);

        // Right boundary
        u(NUM, ind) = ZERO;
        v(NUM + 1, ind) = -v(NUM, ind);

        // Bottom boundary
        u(ind, 0) = -u(ind, 1);
        v(ind, 0) = ZERO;

        // Top boundary
        u(ind, NUM + 1) = TWO - u(ind, NUM);
        v(ind, NUM) = ZERO;

        if (ind == NUM) {
            // Corners
            // Left boundary
            u(0, 0) = ZERO;
            v(0, 0) = -v(1, 0);
            u(0, NUM + 1) = ZERO;
            v(0, NUM + 1) = -v(1, NUM + 1);

            // Right boundary
            u(NUM, 0) = ZERO;
            v(NUM + 1, 0) = -v(NUM, 0);
            u(NUM, NUM + 1) = ZERO;
            v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

            // Bottom boundary
            u(0, 0) = -u(0, 1);
            v(0, 0) = ZERO;
            u(NUM + 1, 0) = -u(NUM + 1, 1);
            v(NUM + 1, 0) = ZERO;

            // Top boundary
            u(0, NUM + 1) = TWO - u(0, NUM);
            v(0, NUM) = ZERO;
            u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
            v(ind, NUM + 1) = ZERO;
        } // end if
    }
} // end set_BCs

// Function to calculate F
__global__ void calculate_F(const Real dt,
                            const Real* __restrict__ u,
                            const Real* __restrict__ v,
                            Real* __restrict__ F,
                            const Real inv_dx2,
                            const Real inv_dy2)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    if (col <= NUM && row <= NUM) {

        if (col == NUM) {
            // Right boundary, F_ij = u_ij
            // Also do left boundary
            F(0, row) = u(0, row);
            F(NUM, row) = u(NUM, row);
        }
        else {

            // u velocities
            Real u_ij = u(col, row);
            Real u_ip1j = u(col + 1, row);
            Real u_ijp1 = u(col, row + 1);
            Real u_im1j = u(col - 1, row);
            Real u_ijm1 = u(col, row - 1);

            // v velocities
            Real v_ij = v(col, row);
            Real v_ip1j = v(col + 1, row);
            Real v_ijm1 = v(col, row - 1);
            Real v_ip1jm1 = v(col + 1, row - 1);

            // Finite differences
            Real du2dx, duvdy, d2udx2, d2udy2;

            Real term1 = u_ij + u_ip1j;
            Real term2 = u_im1j + u_ij;
            du2dx = ((term1 * term1 - term2 * term2)
                + mix_param * (fabs(term1) * (u_ij - u_ip1j)
                    - fabs(term2) * (u_im1j - u_ij))) / (FOUR * dx);

            term1 = v_ij + v_ip1j;
            term2 = v_ijm1 + v_ip1jm1;
            duvdy = ((term1 * (u_ij + u_ijp1) - term2 * (u_ijm1 + u_ij)
                + mix_param * (fabs(term1) * (u_ij - u_ijp1)
                    - fabs(term2) * (u_ijm1 - u_ij)))) / (FOUR * dy);

            d2udx2 = (u_ip1j - (TWO * u_ij) + u_im1j) * inv_dx2;
            d2udy2 = (u_ijp1 - (TWO * u_ij) + u_ijm1) * inv_dy2;

            F(col, row) = u_ij + dt * (((d2udx2 + d2udy2) / Re_num) - du2dx - duvdy + gx);
        } // end if
    }     // end if
} // end calculate_F

// Function to calculate G
__global__ void calculate_G(const Real dt,
                            const Real* __restrict__ u,
                            const Real* __restrict__ v,
                            Real* __restrict__ G,
                            const Real inv_dx2,
                            const Real inv_dy2)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    if (col <= NUM && row <= NUM) {

        if (row == NUM) {
            // Top and bottom boundaries
            G(col, 0) = v(col, 0);
            G(col, NUM) = v(col, NUM);
        }
        else {

            // u velocities
            Real u_ij = u(col, row);
            Real u_ijp1 = u(col, row + 1);
            Real u_im1j = u(col - 1, row);
            Real u_im1jp1 = u(col - 1, row + 1);

            // v velocities
            Real v_ij = v(col, row);
            Real v_ijp1 = v(col, row + 1);
            Real v_ip1j = v(col + 1, row);
            Real v_ijm1 = v(col, row - 1);
            Real v_im1j = v(col - 1, row);

            // Finite differences
            Real dv2dy, duvdx, d2vdx2, d2vdy2;

            Real term1 = v_ij + v_ijp1;
            Real term2 = v_ijm1 + v_ij;
            dv2dy = ((term1 * term1 - term2 * term2)
                + mix_param * (fabs(term1) * (v_ij - v_ijp1)
                    - fabs(term2) * (v_ijm1 - v_ij))) / (FOUR * dy);

            term1 = u_ij + u_ijp1;
            term2 = u_im1j + u_im1jp1;
            duvdx = ((term1 * (v_ij + v_ip1j) - term2 * (v_im1j + v_ij)
                + mix_param * (fabs(term1) * (v_ij - v_ip1j)
                    - fabs(term2) * (v_im1j - v_ij)))) / (FOUR * dx);

            d2vdx2 = (v_ip1j - (TWO * v_ij) + v_im1j) * inv_dx2;
            d2vdy2 = (v_ijp1 - (TWO * v_ij) + v_ijm1) * inv_dy2;

            G(col, row) = v_ij + dt * (((d2vdx2 + d2vdy2) / Re_num) - dv2dy - duvdx + gy);
        } // end if
    }     // end if
} // end calculate_G

// Function to set horizontal pressure boundary conditions
__global__ void set_horz_pres_BCs(Real* __restrict__ pres_red, Real* __restrict__ pres_black)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    col = (col * 2) - 1;

    if (col <= NUM) {
        // int NUM_2 = NUM >> 1;  // Removed local definition of NUM_2

        // p_i,0 = p_i,1
        pres_black(col, 0) = pres_red(col, 1);
        pres_red(col + 1, 0) = pres_black(col + 1, 1);

        // p_i,jmax+1 = p_i,jmax
        pres_red(col, NUM_2 + 1) = pres_black(col, NUM_2);
        pres_black(col + 1, NUM_2 + 1) = pres_red(col + 1, NUM_2);
    }
} // end set_horz_pres_BCs

// Function to set vertical pressure boundary conditions
__global__ void set_vert_pres_BCs(Real* __restrict__ pres_red, Real* __restrict__ pres_black)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

    if (row <= NUM_2) {
        // p_0,j = p_1,j
        pres_black(0, row) = pres_red(1, row);
        pres_red(0, row) = pres_black(1, row);

        // p_imax+1,j = p_imax,j
        pres_black(NUM + 1, row) = pres_red(NUM, row);
        pres_red(NUM + 1, row) = pres_black(NUM, row);
    }
} // end set_vert_pres_BCs

// Function to update pressure for red cells
__global__ void red_kernel(const Real dt,
                           const Real* __restrict__ F,
                           const Real* __restrict__ G,
                           const Real* __restrict__ pres_black,
                           Real* __restrict__ pres_red,
                           const Real inv_dx2,
                           const Real inv_dy2,
                           const Real denom)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    // int NUM_2 = NUM >> 1;  // Removed local definition of NUM_2

    if (col <= NUM && row <= NUM_2) {

        Real p_ij = pres_red(col, row);
        Real p_im1j = pres_black(col - 1, row);
        Real p_ip1j = pres_black(col + 1, row);
        Real p_ijm1 = pres_black(col, row - (col & 1));
        Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

        // Right-hand side
        Real rhs = (((F(col, (2 * row) - (col & 1))
            - F(col - 1, (2 * row) - (col & 1))) / dx)
            + ((G(col, (2 * row) - (col & 1))
                - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

        Real numerator = ((p_ip1j + p_im1j) * inv_dx2) + ((p_ijp1 + p_ijm1) * inv_dy2) - rhs;
        pres_red(col, row) = p_ij * (ONE - omega) + omega * numerator / denom;
    }
} // end red_kernel

// Function to update pressure for black cells
__global__ void black_kernel(const Real dt,
                             const Real* __restrict__ F,
                             const Real* __restrict__ G,
                             const Real* __restrict__ pres_red,
                             Real* __restrict__ pres_black,
                             const Real inv_dx2,
                             const Real inv_dy2,
                             const Real denom)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    // int NUM_2 = NUM >> 1;  // Removed local definition of NUM_2

    if (col <= NUM && row <= NUM_2) {

        Real p_ij = pres_black(col, row);
        Real p_im1j = pres_red(col - 1, row);
        Real p_ip1j = pres_red(col + 1, row);
        Real p_ijm1 = pres_red(col, row - ((col + 1) & 1));
        Real p_ijp1 = pres_red(col, row + (col & 1));

        // Right-hand side
        Real rhs = (((F(col, (2 * row) - ((col + 1) & 1))
            - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
            + ((G(col, (2 * row) - ((col + 1) & 1))
                - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

        Real numerator = ((p_ip1j + p_im1j) * inv_dx2) + ((p_ijp1 + p_ijm1) * inv_dy2) - rhs;
        pres_black(col, row) = p_ij * (ONE - omega) + omega * numerator / denom;
    }
} // end black_kernel

// Kernel to calculate residuals and perform block-wise reduction
__global__ void calc_residual(const Real dt,
                              const Real* __restrict__ F,
                              const Real* __restrict__ G,
                              const Real* __restrict__ pres_red,
                              const Real* __restrict__ pres_black,
                              Real* __restrict__ res)
{
    extern __shared__ Real shared_res[]; // Shared memory for reduction
    int tid = threadIdx.x;

    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    // int NUM_2 = NUM >> 1;  // Removed local definition of NUM_2
    Real res_val = ZERO;

    if (col <= NUM && row <= NUM_2) {

        Real p_ij, p_im1j, p_ip1j, p_ijm1, p_ijp1, rhs, res_local;
        // Red point
        p_ij = pres_red(col, row);

        p_im1j = pres_black(col - 1, row);
        p_ip1j = pres_black(col + 1, row);
        p_ijm1 = pres_black(col, row - (col & 1));
        p_ijp1 = pres_black(col, row + ((col + 1) & 1));

        rhs = (((F(col, (2 * row) - (col & 1)) - F(col - 1, (2 * row) - (col & 1))) / dx)
            + ((G(col, (2 * row) - (col & 1)) - G(col, (2 * row) - (col & 1) - 1)) / dy)) / dt;

        // Calculate residual
        res_local = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
            + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;
        res_val += res_local * res_local;
        // Black point
        p_ij = pres_black(col, row);

        p_im1j = pres_red(col - 1, row);
        p_ip1j = pres_red(col + 1, row);
        p_ijm1 = pres_red(col, row - ((col + 1) & 1));
        p_ijp1 = pres_red(col, row + (col & 1));

        rhs = (((F(col, (2 * row) - ((col + 1) & 1)) - F(col - 1, (2 * row) - ((col + 1) & 1))) / dx)
            + ((G(col, (2 * row) - ((col + 1) & 1)) - G(col, (2 * row) - ((col + 1) & 1) - 1)) / dy)) / dt;

        // Calculate residual
        res_local = ((p_ip1j - (TWO * p_ij) + p_im1j) / (dx * dx))
            + ((p_ijp1 - (TWO * p_ij) + p_ijm1) / (dy * dy)) - rhs;
        res_val += res_local * res_local;
    }

    // Reduce within block
    shared_res[tid] = res_val;
    __syncthreads();

    // Reduce sum in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_res[tid] += shared_res[tid + s];
        }
        __syncthreads();
    }

    // Write result from each block to res array
    if (tid == 0) {
        res[blockIdx.y + (gridDim.y * blockIdx.x)] = shared_res[0];
    }
}

// Kernel to perform global reduction (sum) of residuals
__global__ void reduceSumKernel(Real* g_idata, Real* g_odata, unsigned int n)
{
    extern __shared__ Real sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    Real mySum = ZERO;

    if (i < n) mySum = g_idata[i];
    if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile Real* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Kernel to calculate u velocities and find maximum u velocity
__global__ void calculate_u(const Real dt,
                            const Real* __restrict__ F,
                            const Real* __restrict__ pres_red,
                            const Real* __restrict__ pres_black,
                            Real* __restrict__ u,
                            Real* __restrict__ max_u)
{
    extern __shared__ Real max_cache[]; // Shared memory for reduction
    int tid = threadIdx.x;

    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    Real new_u = ZERO;

    if (col < NUM && row <= NUM) {

        Real p_ij, p_ip1j;
        // Red point
        p_ij = pres_red(col, row);
        p_ip1j = pres_black(col + 1, row);

        new_u = F(col, (2 * row) - (col & 1)) - (dt * (p_ip1j - p_ij) / dx);
        u(col, (2 * row) - (col & 1)) = new_u;

        // Black point
        p_ij = pres_black(col, row);
        p_ip1j = pres_red(col + 1, row);

        Real new_u2 = F(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ip1j - p_ij) / dx);
        u(col, (2 * row) - ((col + 1) & 1)) = new_u2;

        // Check for max of these two
        new_u = fmax(fabs(new_u), fabs(new_u2));
    }
    else if (col == NUM) {
        // Check for maximum velocity in boundary cells
        new_u = fmax(fabs(u(NUM, (2 * row))), fabs(u(0, (2 * row))));
        new_u = fmax(fabs(u(NUM, (2 * row) - 1)), new_u);
        new_u = fmax(fabs(u(0, (2 * row) - 1)), new_u);

        new_u = fmax(fabs(u(NUM + 1, (2 * row))), new_u);
        new_u = fmax(fabs(u(NUM + 1, (2 * row) - 1)), new_u);
    }

    // Store max in shared memory
    max_cache[tid] = new_u;
    __syncthreads();

    // Reduce max in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            max_cache[tid] = fmax(max_cache[tid], max_cache[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile Real* smem = max_cache;
        smem[tid] = fmax(smem[tid], smem[tid + 32]);
        smem[tid] = fmax(smem[tid], smem[tid + 16]);
        smem[tid] = fmax(smem[tid], smem[tid + 8]);
        smem[tid] = fmax(smem[tid], smem[tid + 4]);
        smem[tid] = fmax(smem[tid], smem[tid + 2]);
        smem[tid] = fmax(smem[tid], smem[tid + 1]);
    }

    // Write result from each block to global memory
    if (tid == 0) {
        max_u[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
    }
}

// Kernel to calculate v velocities and find maximum v velocity
__global__ void calculate_v(const Real dt,
                            const Real* __restrict__ G,
                            const Real* __restrict__ pres_red,
                            const Real* __restrict__ pres_black,
                            Real* __restrict__ v,
                            Real* __restrict__ max_v)
{
    extern __shared__ Real max_cache[]; // Shared memory for reduction
    int tid = threadIdx.x;

    int row = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
    int col = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

    Real new_v = ZERO;
    // int NUM_2 = NUM >> 1;  // Removed local definition of NUM_2

    if (row < NUM_2 && col <= NUM) {
        Real p_ij, p_ijp1;

        // Red pressure point
        p_ij = pres_red(col, row);
        p_ijp1 = pres_black(col, row + ((col + 1) & 1));

        new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
        v(col, (2 * row) - (col & 1)) = new_v;

        // Black pressure point
        p_ij = pres_black(col, row);
        p_ijp1 = pres_red(col, row + (col & 1));

        Real new_v2 = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
        v(col, (2 * row) - ((col + 1) & 1)) = new_v2;

        // Check for max of these two
        new_v = fmax(fabs(new_v), fabs(new_v2));
    }
    else if (row == NUM_2) {

        if ((col & 1) == 1) {
            // Black point is on boundary, only calculate red point below it
            Real p_ij = pres_red(col, row);
            Real p_ijp1 = pres_black(col, row + ((col + 1) & 1));

            new_v = G(col, (2 * row) - (col & 1)) - (dt * (p_ijp1 - p_ij) / dy);
            v(col, (2 * row) - (col & 1)) = new_v;
        }
        else {
            // Red point is on boundary, only calculate black point below it
            Real p_ij = pres_black(col, row);
            Real p_ijp1 = pres_red(col, row + (col & 1));

            new_v = G(col, (2 * row) - ((col + 1) & 1)) - (dt * (p_ijp1 - p_ij) / dy);
            v(col, (2 * row) - ((col + 1) & 1)) = new_v;
        }

        // Get maximum v velocity
        new_v = fabs(new_v);

        // Check for maximum velocity in boundary cells also
        new_v = fmax(fabs(v(col, NUM)), new_v);
        new_v = fmax(fabs(v(col, 0)), new_v);
        new_v = fmax(fabs(v(col, NUM + 1)), new_v);
    }

    // Store max in shared memory
    max_cache[tid] = new_v;
    __syncthreads();

    // Reduce max in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            max_cache[tid] = fmax(max_cache[tid], max_cache[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile Real* smem = max_cache;
        smem[tid] = fmax(smem[tid], smem[tid + 32]);
        smem[tid] = fmax(smem[tid], smem[tid + 16]);
        smem[tid] = fmax(smem[tid], smem[tid + 8]);
        smem[tid] = fmax(smem[tid], smem[tid + 4]);
        smem[tid] = fmax(smem[tid], smem[tid + 2]);
        smem[tid] = fmax(smem[tid], smem[tid + 1]);
    }

    // Write result from each block to global memory
    if (tid == 0) {
        max_v[blockIdx.y + (gridDim.y * blockIdx.x)] = max_cache[0];
    }
}

// Kernel to perform global reduction (max) of velocities
__global__ void reduceMaxKernel(Real* g_idata, Real* g_odata, unsigned int n)
{
    extern __shared__ Real sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    Real myMax = ZERO;

    if (i < n) myMax = g_idata[i];
    if (i + blockDim.x < n) myMax = fmax(myMax, g_idata[i + blockDim.x]);

    sdata[tid] = myMax;
    __syncthreads();

    // Reduce max in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile Real* smem = sdata;
        smem[tid] = fmax(smem[tid], smem[tid + 32]);
        smem[tid] = fmax(smem[tid], smem[tid + 16]);
        smem[tid] = fmax(smem[tid], smem[tid + 8]);
        smem[tid] = fmax(smem[tid], smem[tid + 4]);
        smem[tid] = fmax(smem[tid], smem[tid + 2]);
        smem[tid] = fmax(smem[tid], smem[tid + 1]);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char* argv[])
{
    // Iterations for Red-Black Gauss-Seidel with SOR
    int iter = 0;
    const int it_max = 1000000;

    // SOR iteration tolerance
    const Real tol = 0.001;

    // Time range
    const Real time_start = 0.0;
    const Real time_end = 0.001; //20.0;

    // Initial time step size
    Real dt = 0.02;

    int size = (NUM + 2) * (NUM + 2);
    int size_pres = (NUM_2 + 2) * (NUM + 2);

    // Arrays for pressure and velocity
    Real* F;
    Real* u;
    Real* G;
    Real* v;

    // Allocate pinned host memory for faster data transfer
    cudaHostAlloc((void**)&F, size * sizeof(Real), cudaHostAllocDefault);
    cudaHostAlloc((void**)&u, size * sizeof(Real), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G, size * sizeof(Real), cudaHostAllocDefault);
    cudaHostAlloc((void**)&v, size * sizeof(Real), cudaHostAllocDefault);

    // Initialize arrays to ZERO
    memset(F, 0, size * sizeof(Real));
    memset(u, 0, size * sizeof(Real));
    memset(G, 0, size * sizeof(Real));
    memset(v, 0, size * sizeof(Real));

    // Arrays for pressure
    Real* pres_red;
    Real* pres_black;

    cudaHostAlloc((void**)&pres_red, size_pres * sizeof(Real), cudaHostAllocDefault);
    cudaHostAlloc((void**)&pres_black, size_pres * sizeof(Real), cudaHostAllocDefault);

    // Initialize pressure arrays to ZERO
    memset(pres_red, 0, size_pres * sizeof(Real));
    memset(pres_black, 0, size_pres * sizeof(Real));

    // Print problem size
    printf("Problem size: %d x %d \n", NUM, NUM);

    // Block and grid dimensions

    // Boundary conditions kernel
    dim3 block_bcs(BLOCK_SIZE);
    dim3 grid_bcs((NUM + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Pressure kernel
    dim3 block_pr(BLOCK_SIZE, 1);
    dim3 grid_pr((NUM_2 + BLOCK_SIZE - 1) / BLOCK_SIZE, NUM);

    // Block and grid dimensions for F
    dim3 block_F(16, 16);
    dim3 grid_F((NUM + block_F.x - 1) / block_F.x, (NUM + block_F.y - 1) / block_F.y);

    // Block and grid dimensions for G
    dim3 block_G(16, 16);
    dim3 grid_G((NUM + block_G.x - 1) / block_G.x, (NUM + block_G.y - 1) / block_G.y);

    // Horizontal pressure boundary conditions
    dim3 block_hpbc(BLOCK_SIZE);
    dim3 grid_hpbc((NUM_2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Vertical pressure boundary conditions
    dim3 block_vpbc(BLOCK_SIZE);
    dim3 grid_vpbc((NUM_2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Residual variable
    Real* res_d;
    int size_res = grid_pr.x * grid_pr.y;

    // Allocate device memory for residuals
    cudaMalloc((void**)&res_d, size_res * sizeof(Real));

    // Device memory for final residual reduction
    Real* norm_L2_d;
    unsigned int num_blocks_reduce = (size_res + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc((void**)&norm_L2_d, num_blocks_reduce * sizeof(Real));

    // Variables to store maximum velocities
    Real* max_u_d;
    Real* max_v_d;
    int size_max = grid_pr.x * grid_pr.y;

    cudaMalloc((void**)&max_u_d, size_max * sizeof(Real));
    cudaMalloc((void**)&max_v_d, size_max * sizeof(Real));

    // Device memory for final maximum velocity reduction
    Real* max_u_total_d;
    Real* max_v_total_d;
    cudaMalloc((void**)&max_u_total_d, num_blocks_reduce * sizeof(Real));
    cudaMalloc((void**)&max_v_total_d, num_blocks_reduce * sizeof(Real));

    // Set initial BCs
    set_BCs_host(u, v);

    Real max_u = SMALL;
    Real max_v = SMALL;

    // Allocate and transfer device memory
    Real* u_d;
    Real* F_d;
    Real* v_d;
    Real* G_d;

    Real* pres_red_d;
    Real* pres_black_d;

    cudaMalloc((void**)&u_d, size * sizeof(Real));
    cudaMalloc((void**)&F_d, size * sizeof(Real));
    cudaMalloc((void**)&v_d, size * sizeof(Real));
    cudaMalloc((void**)&G_d, size * sizeof(Real));

    cudaMalloc((void**)&pres_red_d, size_pres * sizeof(Real));
    cudaMalloc((void**)&pres_black_d, size_pres * sizeof(Real));

    // Copy to device memory
    cudaMemcpy(u_d, u, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(G_d, G, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(pres_red_d, pres_red, size_pres * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(pres_black_d, pres_black, size_pres * sizeof(Real), cudaMemcpyHostToDevice);

    Real time = time_start;

    // Time-step size based on grid and Reynolds number
    Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Precompute constants
    const Real inv_dx2 = ONE / (dx * dx);
    const Real inv_dy2 = ONE / (dy * dy);
    const Real denom = TWO * (inv_dx2 + inv_dy2);

    // Time iteration loop
    while (time < time_end) {

        // Calculate time step based on stability and CFL
        dt = dt_Re;
        dt = tau * dt;

        if ((time + dt) >= time_end) {
            dt = time_end - time;
        }

        // Calculate F and G
        calculate_F <<<grid_F, block_F>>> (dt, u_d, v_d, F_d, inv_dx2, inv_dy2);
        calculate_G <<<grid_G, block_G>>> (dt, u_d, v_d, G_d, inv_dx2, inv_dy2);

        // Ensure all kernels are finished
        cudaDeviceSynchronize();

        Real norm_L2; // Moved declaration outside the for-loop

        // SOR iteration loop
        for (iter = 1; iter <= it_max; ++iter) {

            // Set pressure boundary conditions
            set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (pres_red_d, pres_black_d);
            set_vert_pres_BCs <<<grid_vpbc, block_vpbc>>> (pres_red_d, pres_black_d);

            // Update red cells
            red_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_black_d, pres_red_d, inv_dx2, inv_dy2, denom);

            // Update black cells
            black_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d, inv_dx2, inv_dy2, denom);

            // Calculate residual values
            calc_residual <<<grid_pr, block_pr, BLOCK_SIZE * sizeof(Real)>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);

            // Perform global reduction on residuals
            reduceSumKernel <<<num_blocks_reduce, BLOCK_SIZE, BLOCK_SIZE * sizeof(Real)>>> (res_d, norm_L2_d, size_res);

            // If necessary, perform further reductions
            unsigned int s = num_blocks_reduce;
            while (s > 1) {
                unsigned int threads = (s > BLOCK_SIZE) ? BLOCK_SIZE : s;
                unsigned int blocks = (s + (threads * 2 - 1)) / (threads * 2);
                reduceSumKernel <<<blocks, threads, threads * sizeof(Real)>>> (norm_L2_d, norm_L2_d, s);
                s = blocks;
            }

            // Copy result back to host
            cudaMemcpy(&norm_L2, norm_L2_d, sizeof(Real), cudaMemcpyDeviceToHost);

            // Calculate residual
            norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM)));

            // If tolerance has been reached, end SOR iterations
            if (norm_L2 < tol) {
                break;
            }
        } // end for

        printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

        // Calculate new velocities and find maximums
        calculate_u <<<grid_pr, block_pr, BLOCK_SIZE * sizeof(Real)>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);

        // Perform global reduction to find max_u
        reduceMaxKernel <<<num_blocks_reduce, BLOCK_SIZE, BLOCK_SIZE * sizeof(Real)>>> (max_u_d, max_u_total_d, size_max);
        unsigned int s = num_blocks_reduce;
        while (s > 1) {
            unsigned int threads = (s > BLOCK_SIZE) ? BLOCK_SIZE : s;
            unsigned int blocks = (s + (threads * 2 - 1)) / (threads * 2);
            reduceMaxKernel <<<blocks, threads, threads * sizeof(Real)>>> (max_u_total_d, max_u_total_d, s);
            s = blocks;
        }
        cudaMemcpy(&max_u, max_u_total_d, sizeof(Real), cudaMemcpyDeviceToHost);

        calculate_v <<<grid_pr, block_pr, BLOCK_SIZE * sizeof(Real)>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);

        // Perform global reduction to find max_v
        reduceMaxKernel <<<num_blocks_reduce, BLOCK_SIZE, BLOCK_SIZE * sizeof(Real)>>> (max_v_d, max_v_total_d, size_max);
        s = num_blocks_reduce;
        while (s > 1) {
            unsigned int threads = (s > BLOCK_SIZE) ? BLOCK_SIZE : s;
            unsigned int blocks = (s + (threads * 2 - 1)) / (threads * 2);
            reduceMaxKernel <<<blocks, threads, threads * sizeof(Real)>>> (max_v_total_d, max_v_total_d, s);
            s = blocks;
        }
        cudaMemcpy(&max_v, max_v_total_d, sizeof(Real), cudaMemcpyDeviceToHost);

        // Set velocity boundary conditions
        set_BCs <<<grid_bcs, block_bcs>>> (u_d, v_d);

        // Increase time
        time += dt;
    } // end while

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);

    // Transfer final velocity values back to host
    cudaMemcpy(u, u_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, v_d, size * sizeof(Real), cudaMemcpyDeviceToHost);

    // Write data to file
    FILE* pfile;
    pfile = fopen("velocity_gpu_refactored.dat", "w");
    fprintf(pfile, "#x\ty\tu\tv\n");
    if (pfile != NULL) {
        for (int row = 0; row < NUM; ++row) {
            for (int col = 0; col < NUM; ++col) {

                Real u_ij = u[(col * NUM) + row];
                Real u_im1j;
                if (col == 0) {
                    u_im1j = 0.0;
                }
                else {
                    u_im1j = u[(col - 1) * NUM + row];
                }

                u_ij = (u_ij + u_im1j) / 2.0;

                Real v_ij = v[(col * NUM) + row];
                Real v_ijm1;
                if (row == 0) {
                    v_ijm1 = 0.0;
                }
                else {
                    v_ijm1 = v[(col * NUM) + row - 1];
                }

                v_ij = (v_ij + v_ijm1) / 2.0;

                fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
            }
        }
    }

    fclose(pfile);

    // Free device memory
    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(F_d);
    cudaFree(G_d);
    cudaFree(pres_red_d);
    cudaFree(pres_black_d);
    cudaFree(max_u_d);
    cudaFree(max_v_d);
    cudaFree(max_u_total_d);
    cudaFree(max_v_total_d);
    cudaFree(norm_L2_d);
    cudaFree(res_d);

    // Free host memory
    cudaFreeHost(pres_red);
    cudaFreeHost(pres_black);
    cudaFreeHost(u);
    cudaFreeHost(v);
    cudaFreeHost(F);
    cudaFreeHost(G);

    return 0;
}
