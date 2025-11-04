
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

/** Problem size along one side; total number of cells is this squared */
#define NUM 512

// block size
#define BLOCK_SIZE 128

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
#define Real double
#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0
#define SMALL 1.0e-10

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
#define SMALL 1.0e-10f

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

// map two-dimensional indices to one-dimensional memory
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM / 2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM / 2) + 2)) + (J)]

// Function and kernel declarations
void set_BCs_host(Real* u, Real* v);
__global__ void calculate_F(Real dt, Real* u, Real* v, Real* F);
__global__ void calculate_G(Real dt, Real* u, Real* v, Real* G);
__global__ void sum_pressure(Real* pres_red, Real* pres_black, Real* pres_sum);
__global__ void set_horz_pres_BCs(Real* pres_red, Real* pres_black);
__global__ void set_vert_pres_BCs(Real* pres_red, Real* pres_black);
__global__ void red_kernel(Real dt, Real* F, Real* G, Real* pres_black, Real* pres_red);
__global__ void black_kernel(Real dt, Real* F, Real* G, Real* pres_red, Real* pres_black);
__global__ void calc_residual(Real dt, Real* F, Real* G, Real* pres_red, Real* pres_black, Real* res);
__global__ void calculate_u(Real dt, Real* F, Real* pres_red, Real* pres_black, Real* u, Real* max_u);
__global__ void calculate_v(Real dt, Real* G, Real* pres_red, Real* pres_black, Real* v, Real* max_v);
__global__ void set_BCs(Real* u, Real* v);

int main(int argc, char *argv[]) {
    // iterations for Red-Black Gauss-Seidel with SOR
    int iter = 0;
    const int it_max = 1000000;

    // SOR iteration tolerance
    const Real tol = 0.001;

    // time range
    const Real time_start = 0.0;
    const Real time_end = 0.001; //20.0;

    // initial time step size
    Real dt = 0.02;

    int size = (NUM + 2) * (NUM + 2);
    int size_pres = ((NUM / 2) + 2) * (NUM + 2);

    // arrays for pressure and velocity
    Real* F;
    Real* u;
    Real* G;
    Real* v;

    F = (Real *) calloc(size, sizeof(Real));
    u = (Real *) calloc(size, sizeof(Real));
    G = (Real *) calloc(size, sizeof(Real));
    v = (Real *) calloc(size, sizeof(Real));

    for (int i = 0; i < size; ++i) {
        F[i] = ZERO;
        u[i] = ZERO;
        G[i] = ZERO;
        v[i] = ZERO;
    }

    // arrays for pressure
    Real* pres_red;
    Real* pres_black;

    pres_red = (Real *) calloc(size_pres, sizeof(Real));
    pres_black = (Real *) calloc(size_pres, sizeof(Real));

    for (int i = 0; i < size_pres; ++i) {
        pres_red[i] = ZERO;
        pres_black[i] = ZERO;
    }

    // print problem size
    printf("Problem size: %d x %d \n", NUM, NUM);

    // block and grid dimensions

    // boundary conditions kernel
    dim3 block_bcs(BLOCK_SIZE, 1);
    dim3 grid_bcs(NUM / BLOCK_SIZE, 1);

    // pressure kernel
    dim3 block_pr(BLOCK_SIZE, 1);
    dim3 grid_pr(NUM / (2 * BLOCK_SIZE), NUM);

    // block and grid dimensions for F
    dim3 block_F(BLOCK_SIZE, 1);
    dim3 grid_F(NUM / BLOCK_SIZE, NUM);

    // block and grid dimensions for G
    dim3 block_G(BLOCK_SIZE, 1);
    dim3 grid_G(NUM / BLOCK_SIZE, NUM);

    // horizontal pressure boundary conditions
    dim3 block_hpbc(BLOCK_SIZE, 1);
    dim3 grid_hpbc(NUM / (2 * BLOCK_SIZE), 1);

    // vertical pressure boundary conditions
    dim3 block_vpbc(BLOCK_SIZE, 1);
    dim3 grid_vpbc(NUM / (2 * BLOCK_SIZE), 1);

    // residual variable
    Real* res;

    int size_res = grid_pr.x * grid_pr.y;
    res = (Real *) calloc(size_res, sizeof(Real));

    // variables to store maximum velocities
    Real* max_u_arr;
    Real* max_v_arr;
    int size_max = grid_pr.x * grid_pr.y;

    max_u_arr = (Real *) calloc(size_max, sizeof(Real));
    max_v_arr = (Real *) calloc(size_max, sizeof(Real));

    // pressure sum
    Real* pres_sum;
    pres_sum = (Real *) calloc(size_res, sizeof(Real));

    // set initial BCs
    set_BCs_host(u, v);

    Real max_u = SMALL;
    Real max_v = SMALL;
    // get max velocity for initial values (including BCs)
    for (int col = 0; col < NUM + 2; ++col) {
        for (int row = 1; row < NUM + 2; ++row) {
            max_u = fmax(max_u, fabs(u(col, row)));
        }
    }

    for (int col = 1; col < NUM + 2; ++col) {
        for (int row = 0; row < NUM + 2; ++row) {
            max_v = fmax(max_v, fabs(v(col, row)));
        }
    }

    // allocate and transfer device memory
    Real* u_d;
    Real* F_d;
    Real* v_d;
    Real* G_d;

    Real* pres_red_d;
    Real* pres_black_d;
    Real* pres_sum_d;
    Real* res_d;

    Real* max_u_d;
    Real* max_v_d;

    cudaMalloc((void**) &u_d, size * sizeof(Real));
    cudaMalloc((void**) &F_d, size * sizeof(Real));
    cudaMalloc((void**) &v_d, size * sizeof(Real));
    cudaMalloc((void**) &G_d, size * sizeof(Real));

    cudaMalloc((void**) &pres_red_d, size_pres * sizeof(Real));
    cudaMalloc((void**) &pres_black_d, size_pres * sizeof(Real));

    cudaMalloc((void**) &pres_sum_d, size_res * sizeof(Real));
    cudaMalloc((void**) &res_d, size_res * sizeof(Real));
    cudaMalloc((void**) &max_u_d, size_max * sizeof(Real));
    cudaMalloc((void**) &max_v_d, size_max * sizeof(Real));

    // copy to device memory
    cudaMemcpy(u_d, u, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(G_d, G, size * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(pres_red_d, pres_red, size_pres * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(pres_black_d, pres_black, size_pres * sizeof(Real), cudaMemcpyHostToDevice);

    Real time = time_start;

    // time-step size based on grid and Reynolds number
    Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // time iteration loop
    while (time < time_end) {

        // calculate time step based on stability and CFL
        dt = fmin((dx / max_u), (dy / max_v));
        dt = tau * fmin(dt_Re, dt);

        if ((time + dt) >= time_end) {
            dt = time_end - time;
        }

        // calculate F and G
        calculate_F <<<grid_F, block_F>>> (dt, u_d, v_d, F_d);
        calculate_G <<<grid_G, block_G>>> (dt, u_d, v_d, G_d);

        // get L2 norm of initial pressure
        sum_pressure <<<grid_pr, block_pr>>> (pres_red_d, pres_black_d, pres_sum_d);
        cudaMemcpy(pres_sum, pres_sum_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost);

        Real p0_norm = ZERO;
        for (int i = 0; i < size_res; ++i) {
            p0_norm += pres_sum[i];
        }

        p0_norm = sqrt(p0_norm / ((Real)(NUM * NUM)));
        if (p0_norm < 0.0001) {
            p0_norm = 1.0;
        }

        Real norm_L2;

        // calculate new pressure
        // red-black Gauss-Seidel with SOR iteration loop
        for (iter = 1; iter <= it_max; ++iter) {

            // set pressure boundary conditions
            set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (pres_red_d, pres_black_d);
            set_vert_pres_BCs <<<grid_vpbc, block_vpbc>>> (pres_red_d, pres_black_d);

            // update red cells
            red_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_black_d, pres_red_d);

            // update black cells
            black_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d);

            // calculate residual values
            calc_residual <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);

            // transfer residual value(s) back to CPU
            cudaMemcpy(res, res_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost);

            norm_L2 = ZERO;
            for (int i = 0; i < size_res; ++i) {
                norm_L2 += res[i];
            }

            // calculate residual
            norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

            // if tolerance has been reached, end SOR iterations
            if (norm_L2 < tol) {
                break;
            }
        } // end for

        printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

        // calculate new velocities and transfer maximums back
        calculate_u <<<grid_pr, block_pr>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
        calculate_v <<<grid_pr, block_pr>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);

        // Use thrust library to perform reduction on the device for max_u and max_v
        thrust::device_ptr<Real> max_u_ptr(max_u_d);
        thrust::device_ptr<Real> max_v_ptr(max_v_d);

        max_u = thrust::reduce(max_u_ptr, max_u_ptr + size_max, ZERO, thrust::maximum<Real>());
        max_v = thrust::reduce(max_v_ptr, max_v_ptr + size_max, ZERO, thrust::maximum<Real>());

        // set velocity boundary conditions
        set_BCs <<<grid_bcs, block_bcs>>> (u_d, v_d);

        // increase time
        time += dt;

    } // end while

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);

    // transfer final temperature values back
    cudaMemcpy(u, u_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, v_d, size * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(pres_red, pres_red_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(pres_black, pres_black_d, size_pres * sizeof(Real), cudaMemcpyDeviceToHost);

    // write data to file
    FILE * pfile;
    pfile = fopen("velocity_gpu.dat", "w");
    fprintf(pfile, "#x\ty\tu\tv\n");
    if (pfile != NULL) {
        for (int row = 0; row < NUM; ++row) {
            for (int col = 0; col < NUM; ++col) {

                Real u_ij = u[(col * NUM) + row];
                Real u_im1j;
                if (col == 0) {
                    u_im1j = 0.0;
                } else {
                    u_im1j = u[(col - 1) * NUM + row];
                }

                u_ij = (u_ij + u_im1j) / 2.0;

                Real v_ij = v[(col * NUM) + row];
                Real v_ijm1;
                if (row == 0) {
                    v_ijm1 = 0.0;
                } else {
                    v_ijm1 = v[(col * NUM) + row - 1];
                }

                v_ij = (v_ij + v_ijm1) / 2.0;

                fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
            }
        }
    }

    fclose(pfile);

    // free device memory
    cudaFree(u_d);
    cudaFree(v_d);
    cudaFree(F_d);
    cudaFree(G_d);
    cudaFree(pres_red_d);
    cudaFree(pres_black_d);
    cudaFree(max_u_d);
    cudaFree(max_v_d);
    cudaFree(pres_sum_d);
    cudaFree(res_d);

    free(pres_red);
    free(pres_black);
    free(u);
    free(v);
    free(F);
    free(G);
    free(max_u_arr);
    free(max_v_arr);
    free(res);
    free(pres_sum);

    return 0;
}

// Implementations of the declared functions and kernels should be provided here.

void set_BCs_host(Real* u, Real* v) {
    // Implementation of set_BCs_host function
}

__global__ void calculate_F(Real dt, Real* u, Real* v, Real* F) {
    // Implementation of calculate_F kernel
}

__global__ void calculate_G(Real dt, Real* u, Real* v, Real* G) {
    // Implementation of calculate_G kernel
}

__global__ void sum_pressure(Real* pres_red, Real* pres_black, Real* pres_sum) {
    // Implementation of sum_pressure kernel
}

__global__ void set_horz_pres_BCs(Real* pres_red, Real* pres_black) {
    // Implementation of set_horz_pres_BCs kernel
}

__global__ void set_vert_pres_BCs(Real* pres_red, Real* pres_black) {
    // Implementation of set_vert_pres_BCs kernel
}

__global__ void red_kernel(Real dt, Real* F, Real* G, Real* pres_black, Real* pres_red) {
    // Implementation of red_kernel
}

__global__ void black_kernel(Real dt, Real* F, Real* G, Real* pres_red, Real* pres_black) {
    // Implementation of black_kernel
}

__global__ void calc_residual(Real dt, Real* F, Real* G, Real* pres_red, Real* pres_black, Real* res) {
    // Implementation of calc_residual kernel
}

__global__ void calculate_u(Real dt, Real* F, Real* pres_red, Real* pres_black, Real* u, Real* max_u) {
    // Implementation of calculate_u kernel
}

__global__ void calculate_v(Real dt, Real* G, Real* pres_red, Real* pres_black, Real* v, Real* max_v) {
    // Implementation of calculate_v kernel
}

__global__ void set_BCs(Real* u, Real* v) {
    // Implementation of set_BCs kernel
}
