
//  1D Helmholtz FEM: -u'' + u = f,  u(0)=1,  u'(1)=0
//  Exact solution: u = cos(2*pi*x)
//
//  All matrices stored as 1D vectors (row-major)
//  A[i][j]  →  A[i*n + j]
//
//  Compile:  g++ -O2 -o fem1d_flat fem1d_flat.cpp
//------------------------------------------------------------


// 0504: port the CPU code to the GPU version:


// 0510: use the cuSparse + cpoy only 3 diagonals to GPU for the linear solver, instead of the full (M+L) matrix:




#include <iostream>
#include <vector>
#include <cmath>
// use the Eigen library for dense linear algebra 
#include <Eigen/Dense>

#include <iostream>
#include <fstream>


#include <cusparse.h>
#include <cusolverDn.h>    

#include <fstream>

#include <chrono>
using namespace std::chrono;





#define CUDA_CHECK(call) { cudaError_t err = call; if(err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

using namespace std;

const double PI = acos(-1.0);

// helper: index into a flat n x n matrix
__host__ __device__ inline int idx(int i, int j, int n) { return i * n + j; }


__device__ __constant__ double PI_d = 3.14159265358979323846;

//------------------------------------------------------------
// STEP 1: element matrices for ONE element
// Me, Le are flat 2x2 → size 4
// fe is size 2
//------------------------------------------------------------


// call this form GPU:

__device__ void compute_element_matrices(double dx, double x_left,
                                          double Me[4],
                                          double Le[4],
                                          double fe[2])
{
    double ksai[2] = { sqrt(3.0)/3.0, -sqrt(3.0)/3.0 };
    double Je      = dx / 2.0;

    double phi[4];
    phi[0] = (1.0 + ksai[0]) / 2.0;
    phi[1] = (1.0 + ksai[1]) / 2.0;
    phi[2] = (1.0 - ksai[0]) / 2.0;
    phi[3] = (1.0 - ksai[1]) / 2.0;

    double dphi[2] = { -0.5, 0.5 };

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            Me[i*2+j] = Je * (phi[i*2+0]*phi[j*2+0]
                             + phi[i*2+1]*phi[j*2+1]);

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            Le[i*2+j] = (2.0/Je) * dphi[i] * dphi[j];

    double x_g[2];
    x_g[0] = x_left + Je * (1.0 + ksai[0]);
    x_g[1] = x_left + Je * (1.0 + ksai[1]);

    double f_g[2];
        f_g[0] = (4.0*PI_d*PI_d + 1.0) * cos(2.0*PI_d*x_g[0]);
        f_g[1] = (4.0*PI_d*PI_d + 1.0) * cos(2.0*PI_d*x_g[1]);
    for (int i = 0; i < 2; i++)
        fe[i] = Je * (phi[i*2+0]*f_g[0] + phi[i*2+1]*f_g[1]);
}


//------------------------------------------------------------
// STEP 2: assemble global matrices
// M_global, L_global: flat (n_nodes x n_nodes)
// F_global: size n_nodes
//------------------------------------------------------------
// void assemble(int n_elem, double dx,
//               std::vector<double>& M_global,
//               std::vector<double>& L_global,
//               std::vector<double>& F_global)
// {
//     int n_nodes = n_elem + 1;

//     // zero out
//     std::fill(M_global.begin(), M_global.end(), 0.0);
//     std::fill(L_global.begin(), L_global.end(), 0.0);
//     std::fill(F_global.begin(), F_global.end(), 0.0);

//     for (int e = 0; e < n_elem; e++) {

//         // connectivity: local node 0 → global e
//         //               local node 1 → global e+1
//         int conn[2] = { e, e+1 };

//         double x_left = e * dx;

//         double Me[4], Le[4], fe[2];
//         compute_element_matrices(dx, x_left, Me, Le, fe);

//         // scatter-add into global matrices
//         for (int i = 0; i < 2; i++) {
//             int gi = conn[i];
//             F_global[gi] += fe[i];
//             for (int j = 0; j < 2; j++) {
//                 int gj = conn[j];
//                 M_global[idx(gi,gj,n_nodes)] += Me[i*2+j];
//                 L_global[idx(gi,gj,n_nodes)] += Le[i*2+j];
//             }
//         }

// //         i=0: gi = conn[0] = 0
// //      j=0: gj = conn[0] = 0  →  M_global[0,0] += Me[0,0]
// //      j=1: gj = conn[1] = 1  →  M_global[0,1] += Me[0,1]

// //         i=1: gi = conn[1] = 1
// //      j=0: gj = conn[0] = 0  →  M_global[1,0] += Me[1,0]
// //      j=1: gj = conn[1] = 1  →  M_global[1,1] += Me[1,1]




//     }
// }

__global__ void assemble(int n_elem, double dx,
              double* dM_global,
              double* dL_global,
              double* dF_global){

                int n_nodes = n_elem + 1;
                // create memory for element matrices:
                // std::fill(M_global.begin(), M_global.end(), 0.0);
                // std::fill(L_global.begin(), L_global.end(), 0.0);
                // std::fill(F_global.begin(), F_global.end(), 0.0);

               

                int e = blockIdx.x * blockDim.x + threadIdx.x;  // this inastead of loop for each element:
                if (e >= n_elem) return;

                // connectivity: local node 0 → global e
                //               local node 1 → global e+1
                int conn[2] = { e, e+1 };
                double x_left = e * dx;
                double Me[4], Le[4], fe[2];
                compute_element_matrices(dx, x_left, Me, Le, fe);

                // scatter-add into global matrices
                for (int i = 0; i < 2; i++) {
                    int gi = conn[i];
                    atomicAddDouble(&dF_global[gi], fe[i]);
                    //atomicAdd(&dF_global[gi], fe[i]);
                    for (int j = 0; j < 2; j++) {
                        int gj = conn[j];
                        // atomicAdd(&dM_global[idx(gi,gj,n_nodes)], Me[i*2+j]);
                        // atomicAdd(&dL_global[idx(gi,gj,n_nodes)], Le[i*2+j]);





                        
                        atomicAddDouble(&dM_global[idx(gi,gj,n_nodes)], Me[i*2+j]);
                        atomicAddDouble(&dL_global[idx(gi,gj,n_nodes)], Le[i*2+j]);


                    }
                }


              }

//------------------------------------------------------------
// STEP 3: apply Dirichlet BC at node 0 (u=1)
// Operates on flat (n_nodes x n_nodes) matrices
// Shrinks system to (n_elem x n_elem) in-place
//------------------------------------------------------------
void apply_dirichlet(int n_elem,
                     std::vector<double>& M,
                     std::vector<double>& L,
                     std::vector<double>& F)
{
    int n = n_elem + 1;   // current size
    double u0 = 1.0;

    // subtract col 0 from RHS
    for (int i = 1; i < n; i++)
        F[i] -= (M[idx(i,0,n)] + L[idx(i,0,n)]) * u0;

    // shrink: copy rows/cols 1..n-1 into positions 0..n-2
    for (int i = 0; i < n_elem; i++) {
        F[i] = F[i+1];
        for (int j = 0; j < n_elem; j++) {
            M[idx(i,j,n_elem)] = M[idx(i+1,j+1,n)];
            L[idx(i,j,n_elem)] = L[idx(i+1,j+1,n)];
        }
    }
}

//------------------------------------------------------------
// STEP 4: solve (M+L)u = F  via Gauss-Jordan elimination
// A is flat (n x n), F is size n
//------------------------------------------------------------
std::vector<double> solve(int n,
                           std::vector<double>& M,
                           std::vector<double>& L,
                           std::vector<double>& F)
{
    // build augmented matrix [M+L | F], flat row-major
    // size: n x (n+1)
    int ncols = n + 1;
    std::vector<double> A(n * ncols);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            A[i*ncols + j] = M[idx(i,j,n)] + L[idx(i,j,n)];
        A[i*ncols + n] = F[i];
    }

    // Gauss-Jordan with partial pivoting
    for (int col = 0; col < n; col++) {

        // find pivot row
        int pivot = col;
        for (int row = col+1; row < n; row++)
            if (fabs(A[row*ncols+col]) > fabs(A[pivot*ncols+col]))
                pivot = row;

        // swap rows
        for (int j = 0; j <= n; j++)
            std::swap(A[col*ncols+j], A[pivot*ncols+j]);

        // normalise pivot row
        double diag = A[col*ncols+col];
        for (int j = col; j <= n; j++)
            A[col*ncols+j] /= diag;

        // eliminate column
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = A[row*ncols+col];
            for (int j = col; j <= n; j++)
                A[row*ncols+j] -= factor * A[col*ncols+j];
        }
    }

    // extract solution from last column
    std::vector<double> u(n);
    for (int i = 0; i < n; i++)
        u[i] = A[i*ncols + n];
    return u;
}





std::vector<double> solveEigen(int n,
                           std::vector<double>& M,
                           std::vector<double>& L,
                           std::vector<double>& F)
{
    // build Eigen matrix and vector
    Eigen::MatrixXd A(n, n);
    Eigen::VectorXd b(n);

    for (int i = 0; i < n; i++) {
        b(i) = F[i];
        for (int j = 0; j < n; j++)
            A(i,j) = M[idx(i,j,n)] + L[idx(i,j,n)];
    }

    // solve — same as MATLAB: u = A \ b
    Eigen::VectorXd u = A.lu().solve(b);

    // copy back to std::vector
    std::vector<double> result(n);
    for (int i = 0; i < n; i++)
        result[i] = u(i);
    return result;
}



// for thr linear solver on GPU: 
//std::vector<double> u_inner=solveEigen(n_elem,M_global, L_global, F_global);

std::vector<double> solveGPU(int n,
                           std::vector<double>& M,
                           std::vector<double>& L,
                           std::vector<double>& F,
                           cusolverDnHandle_t handle )   
{
    // build A = M + L on CPU
    std::vector<double> A(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*n+j] = M[idx(i,j,n)] + L[idx(i,j,n)];

   // allocate GPU memory
    double *dA, *dF;
    int *dPivot, *dInfo;
    cudaMalloc(&dA,     n*n*sizeof(double));
    cudaMalloc(&dF,     n*sizeof(double));
    cudaMalloc(&dPivot, n*sizeof(int));
    cudaMalloc(&dInfo,  sizeof(int));

    // copy A and F to GPU
    cudaMemcpy(dA, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dF, F.data(), n*sizeof(double),   cudaMemcpyHostToDevice);
    
    



    // query workspace size
    int lwork;
    cusolverDnDgetrf_bufferSize(handle, n, n, dA, n, &lwork);

    double *dWork;
    cudaMalloc(&dWork, lwork*sizeof(double));

    // LU factorisation

    //does the LU factorisation (like MATLAB's lu())
    cusolverDnDgetrf(handle, n, n, dA, n, dWork, dPivot, dInfo);

    // solve
    //does the solve using the factorisation (like MATLAB's \)
    cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, dA, n, dPivot, dF, n, dInfo);

    // copy result back
    std::vector<double> result(n);
    cudaMemcpy(result.data(), dF, n*sizeof(double), cudaMemcpyDeviceToHost);

    // cleanup
   
    cudaFree(dA); cudaFree(dF); cudaFree(dPivot);
    cudaFree(dInfo); cudaFree(dWork);

    return result;
    

}



// 0510: for linear solver: use cuSparse:


std::vector<double> solveGPU_sparse(int n,
                                    std::vector<double>& dl,   // lower diagonal, length n-1
                                    std::vector<double>& d,    // main diagonal,  length n
                                    std::vector<double>& du,   // upper diagonal, length n-1
                                    std::vector<double>& F,    // RHS,            length n
                                    cusparseHandle_t handle)
{
    // ── 1. Pad lower/upper to length n (gtsv2 expects all three of length n) ──
    std::vector<double> dl_pad(n, 0.0), du_pad(n, 0.0);
    std::copy(dl.begin(), dl.end(), dl_pad.begin() + 1);  // dl[0]   = 0 (unused)
    std::copy(du.begin(), du.end(), du_pad.begin());       // du[n-1] = 0 (unused)

    // ── 2. Allocate GPU memory ────────────────────────────────────────────────
    double *dDl, *dD, *dDu, *dF;
    cudaMalloc(&dDl, n * sizeof(double));  // cpu: dl_pad.data() → gpu: dDl
    cudaMalloc(&dD,  n * sizeof(double));  // cpu: d.data()      → gpu: dD
    cudaMalloc(&dDu, n * sizeof(double));  // cpu: du_pad.data() → gpu: dDu
    cudaMalloc(&dF,  n * sizeof(double));  // cpu: F.data()      → gpu: dF

    // ── 3. Copy to GPU ────────────────────────────────────────────────────────
    cudaMemcpy(dDl, dl_pad.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dD,  d.data(),      n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dDu, du_pad.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dF,  F.data(),      n * sizeof(double), cudaMemcpyHostToDevice);

    // ── 4. Query workspace size ───────────────────────────────────────────────
    size_t bufferSize;
    cusparseDgtsv2_bufferSizeExt(handle, n, 1, dDl, dD, dDu, dF, n, &bufferSize);

    void* dBuffer;
    cudaMalloc(&dBuffer, bufferSize);

    // ── 5. Solve  (overwrites dF with the solution x) ─────────────────────────

    // cuSparse:
    cusparseDgtsv2(handle, n, 1, dDl, dD, dDu, dF, n, dBuffer);

    // cuSolver:
    //cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, dA, n, dPivot, dF, n, dInfo);

    // ── 6. Copy result back ───────────────────────────────────────────────────
    std::vector<double> result(n);
    cudaMemcpy(result.data(), dF, n * sizeof(double), cudaMemcpyDeviceToHost);

    // ── 7. Cleanup ────────────────────────────────────────────────────────────
    cudaFree(dDl); cudaFree(dD); cudaFree(dDu);
    cudaFree(dF);  cudaFree(dBuffer);

    return result;
}


//------------------------------------------------------------
// STEP 5: compute errors vs exact solution
//------------------------------------------------------------
void compute_errors(const std::vector<double>& u,
                    const std::vector<double>& x,
                    double& L2, double& Linf)
{
    int n = u.size();
    L2 = 0.0; Linf = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(u[i] - cos(2.0*PI*x[i]));
        L2   += err * err;
        Linf  = std::max(Linf, err);
    }
    L2 = sqrt(L2 / n);
}



// 0510: extract dl, d, du from M_global+L_global for the tridiagonal solver:
//  extract_diagonals_and_F<<<blocksPerGrid, threadsPerBlock>>>(
//         n_elem, dM_global, dL_global, dF_global,
//         d_diag, d_lower, d_upper, d_F_inner);

__global__ void extract_diagonals_and_F(int n_elem,
                            double* dM_global,
                            double* dL_global,
                            double* dF_global,
                            double* d_diag,
                            double* d_lower,
                            double* d_upper,
                            double* d_F_inner)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) return;

    int n_nodes = n_elem + 1;
    int row = i + 1;   // skip Dirichlet node 0

    // main diagonal — every thread writes this
    d_diag[i] = dM_global[idx(row, row, n_nodes)] + dL_global[idx(row, row, n_nodes)];

    // lower diagonal — guard against i=0
    if (i > 0)
        d_lower[i-1] = dM_global[idx(row, row-1, n_nodes)] + dL_global[idx(row, row-1, n_nodes)];

    // upper diagonal — guard against i=n_elem-1
    if (i < n_elem-1)
        d_upper[i] = dM_global[idx(row, row+1, n_nodes)] + dL_global[idx(row, row+1, n_nodes)];

    // RHS with Dirichlet BC subtracted at i=0 only
    double bc = (i == 0) ? (dM_global[idx(1, 0, n_nodes)] + dL_global[idx(1, 0, n_nodes)]) * 1.0
                         : 0.0;
    d_F_inner[i] = dF_global[row] - bc;
}

//------------------------------------------------------------
// MAIN
//------------------------------------------------------------
int main()
{



    //  std::ofstream f1("timing_LSCPU.txt");
    // f1 << "n\ttime\n";

    std::ofstream f2("timing_LSGPU_cuSparseOptimise.txt");
    f2 << "n\ttime\n";



    // warmup - forces CUDA context initialisation
        cudaSetDevice(0);
        double *d_dummy;
        cudaMalloc(&d_dummy, sizeof(double));
        cudaFree(d_dummy);


    // test GPU is working:
        double test_val = 42.0;
        double *d_test, h_test = 0.0;
        CUDA_CHECK(cudaMalloc(&d_test, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_test, &test_val, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&h_test, d_test, sizeof(double), cudaMemcpyDeviceToHost));
        cout << "GPU test: sent 42, got " << h_test << endl;
        cudaFree(d_test);



    std::vector<int> n_list = {100, 1000, 10000};

    std::cout << "n\tL2 error\t\tLinf error\n";
    std::cout << "------------------------------------------------\n";


    // create cuSOLVER handle
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);


    // create cuSparse handle

    cusparseHandle_t sparseHandle;
    cusparseCreate(&sparseHandle);
    

    for (int n_elem : n_list) {

        double dx     = 1.0 / n_elem;
        int n_nodes   = n_elem + 1;

        // node coordinates
        std::vector<double> x(n_nodes);
        for (int i = 0; i < n_nodes; i++)
            x[i] = i * dx;


    
       
        auto t1 = high_resolution_clock::now();
       
        // (1) calculate Me, Le and Fe and assemble on GPU:

        double* dM_global, *dL_global, *dF_global;
        cudaMalloc(&dM_global, n_nodes*n_nodes*sizeof(double));
        cudaMalloc(&dL_global, n_nodes*n_nodes*sizeof(double));
        cudaMalloc(&dF_global, n_nodes*sizeof(double));
        cudaMemset(dM_global, 0, n_nodes * n_nodes * sizeof(double));
        cudaMemset(dL_global, 0, n_nodes * n_nodes * sizeof(double));
        cudaMemset(dF_global, 0, n_nodes * sizeof(double));

         // launch kernel with enough threads to cover all elements
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_elem + threadsPerBlock - 1) / threadsPerBlock;

        assemble<<<blocksPerGrid, threadsPerBlock>>>(n_elem, dx, dM_global, dL_global, dF_global);
        cudaDeviceSynchronize();   // ← wait for kernel to finish


        // copy from GPU back to CPU:

        // allocate CPU vectors
            // std::vector<double> M_global(n_nodes*n_nodes);
            // std::vector<double> L_global(n_nodes*n_nodes);
            // std::vector<double> F_global(n_nodes);

            // // copy GPU → CPU
            // cudaMemcpy(M_global.data(), dM_global, n_nodes*n_nodes*sizeof(double), cudaMemcpyDeviceToHost);
            // cudaMemcpy(L_global.data(), dL_global, n_nodes*n_nodes*sizeof(double), cudaMemcpyDeviceToHost);
            // cudaMemcpy(F_global.data(), dF_global, n_nodes*sizeof(double),         cudaMemcpyDeviceToHost);





            // // (3) apply the boundary condition on CPU:

            // apply_dirichlet(n_elem, M_global, L_global, F_global);
       // std::vector<double> u_inner = solve(n_elem, M_global, L_global, F_global);


        // (1) use Eigen to solver the (M+L)u=F system
        //std::vector<double> u_inner=solveEigen(n_elem,M_global, L_global, F_global);

        // use Eigen in GPU version:
        // std::vector<double> solveGPU(int n,
        //                    std::vector<double>& M,
        //                    std::vector<double>& L,
        //                    std::vector<double>& F)

        // (2) use cuSOLVER to solve the (M+L)u=F system on GPU:

        //std::vector<double> u_inner=solveGPU(n_elem,M_global, L_global, F_global, handle);

        // 0510: (3) use cuSparse to solve the (M+L)u=F system on GPU:

        // solveGPU_sparse(int n,
        //                             std::vector<double>& dl,   // lower diagonal, length n-1
        //                             std::vector<double>& d,    // main diagonal,  length n
        //                             std::vector<double>& du,   // upper diagonal, length n-1
        //                             std::vector<double>& F,    // RHS,            length n


        //                             cusparseHandle_t handle)

         //std::vector<double> u_inner=solveGPU_sparse(n_elem,M_global, L_global, F_global, handle);

         // get dl, d, du from M_global+L_global for the tridiagonal solver:
        std::vector<double> dl(n_elem-1), d(n_elem), du(n_elem-1), F(n_elem);

    //    for (int i = 0; i < n_elem; i++) {

    //             d[i] = M_global[idx(i, i, n_elem)] + L_global[idx(i, i, n_elem)];

    //             if (i > 0) {
    //                 dl[i-1] = M_global[idx(i, i-1, n_elem)] + L_global[idx(i, i-1, n_elem)];
    //             }

    //             if (i < n_elem-1) {
    //                 du[i] = M_global[idx(i, i+1, n_elem)] + L_global[idx(i, i+1, n_elem)];
    //             }

    //             F[i] = F_global[i];
    //    }
        // solveGPU_sparse(int n,
        //                             std::vector<double>& dl,   // lower diagonal, length n-1
        //                             std::vector<double>& d,    // main diagonal,  length n
        //                             std::vector<double>& du,   // upper diagonal, length n-1
        //                             std::vector<double>& F,    // RHS,            length n


        //                             cusparseHandle_t handle)

       // 0510: extract dl, d, du from M_global+L_global for the tridiagonal solver:

       double *d_diag, *d_lower, *d_upper, *d_F_inner;
    cudaMalloc(&d_diag,    n_elem     * sizeof(double));
    cudaMalloc(&d_lower,  (n_elem-1)  * sizeof(double));
    cudaMalloc(&d_upper,  (n_elem-1)  * sizeof(double));
    cudaMalloc(&d_F_inner, n_elem     * sizeof(double));
        extract_diagonals_and_F<<<blocksPerGrid, threadsPerBlock>>>(
        n_elem, dM_global, dL_global, dF_global,
        d_diag, d_lower, d_upper, d_F_inner);
        cudaDeviceSynchronize();

        // only copy 3 small arrays (~0.24 MB for n=10000)
        cudaMemcpy(d.data(),  d_diag,    n_elem*sizeof(double),     cudaMemcpyDeviceToHost);
        cudaMemcpy(dl.data(), d_lower,   (n_elem-1)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(du.data(), d_upper,   (n_elem-1)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(F.data(),  d_F_inner, n_elem*sizeof(double),     cudaMemcpyDeviceToHost);





        // add after cudaMemcpy of diagonals back to CPU
        printf("\n=== Single GPU diagonal check ===\n");
        printf("d[0]    = %.10f\n", d[0]);
        printf("d[9999] = %.10f\n", d[9999]);
        printf("dl[0]   = %.10f\n", dl[0]);
        printf("du[0]   = %.10f\n", du[0]);
        printf("F[0]    = %.10f\n", F[0]);
        printf("F[9999] = %.10f\n", F[9999]);


       
        std::vector<double> u_inner=solveGPU_sparse(n_elem,dl, d, du,F, sparseHandle);


        auto t2 = high_resolution_clock::now();
        double t = duration<double>(t2-t1).count();
        f2 << n_elem << "\t" << t << "\n";


        // restore full solution with u(0)=1
        std::vector<double> u(n_nodes);
        u[0] = 1.0;
        for (int i = 0; i < n_elem; i++)
            u[i+1] = u_inner[i];
        


        // output u to check:

              ofstream file110("uFluid.txt",ios::app);

            for(int i=0;i<n_nodes;i++){
               

                    file110  << u[i] <<endl;
            }

        // errors
        double L2, Linf;
        compute_errors(u, x, L2, Linf);

        std::cout << n_elem << "\t" << L2 << "\t\t" << Linf << "\n";

       
        cudaFree(dM_global);   
        cudaFree(dL_global);   
        cudaFree(dF_global);   
        cudaFree(d_diag);
        cudaFree(d_lower);
        cudaFree(d_upper);
        cudaFree(d_F_inner);
        
    }
    cusolverDnDestroy(handle);
    cusparseDestroy(sparseHandle);  // cleanup cuSparse handle

    return 0;
}
