
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

// 0514: change from single GPU to multi GPU;




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

__global__ void assemble(int n_elem_local,    // elements this GPU handles
                          int n_elem_total,    // total elements across both GPUs
                          double dx,
                          double* dM_global,
                          double* dL_global,
                          double* dF_global,
                          int offset)          // GPU 0: 0, GPU 1: n_elem_total/2
{
    int n_nodes = n_elem_total + 1;            // global matrix size

    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem_local) return;             // local guard

    int e_global = e + offset;                 // global element index

    int conn[2] = { e_global, e_global+1 };
    double x_left = e_global * dx;
    double Me[4], Le[4], fe[2];
    compute_element_matrices(dx, x_left, Me, Le, fe);

    for (int i = 0; i < 2; i++) {
        int gi = conn[i];
        atomicAddDouble(&dF_global[gi], fe[i]);
        for (int j = 0; j < 2; j++) {
            int gj = conn[j];
            atomicAddDouble(&dM_global[idx(gi, gj, n_nodes)], Me[i*2+j]);
            atomicAddDouble(&dL_global[idx(gi, gj, n_nodes)], Le[i*2+j]);
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


    // define all the parameters:
    int n_elem_total = 1000;
    int n_elem_local = n_elem_total / 2;
    int n_nodes_global = n_elem_total + 1;




    // warmup - forces CUDA context initialisation
        cudaSetDevice(0);
      

    // test GPU is working:
        double test_val = 42.0;
        double *d_test, h_test = 0.0;
        CUDA_CHECK(cudaMalloc(&d_test, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_test, &test_val, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&h_test, d_test, sizeof(double), cudaMemcpyDeviceToHost));
        cout << "GPU test: sent 42, got " << h_test << endl;
        cudaFree(d_test);



    std::vector<int> n_list = {1000};

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
        std::vector<double> x(n_nodes_global);
        for (int i = 0; i < n_nodes_global; i++)
            x[i] = i * dx;


        auto t1 = high_resolution_clock::now();


        // (1) for GPU0:
        double* dM_global0, *dL_global0, *dF_global0;
        cudaSetDevice(0);   



        // cudaError_t err;
        // err = cudaMalloc(&dM_global0, n_nodes_global * n_nodes_global * sizeof(double));
        // printf("GPU0 dM malloc: %s\n", cudaGetErrorString(err));
        // err = cudaMalloc(&dL_global0, n_nodes_global * n_nodes_global * sizeof(double));
        // printf("GPU0 dL malloc: %s\n", cudaGetErrorString(err));
        // err = cudaMalloc(&dF_global0, n_nodes_global * sizeof(double));
        // printf("GPU0 dF malloc: %s\n", cudaGetErrorString(err));

        cudaMalloc(&dM_global0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dL_global0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dF_global0, n_nodes_global * sizeof(double));


        cudaMemset(dM_global0, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dL_global0, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dF_global0, 0, n_nodes_global * sizeof(double));

        // call the assemble;

        // launch kernel with enough threads to cover all elements
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_elem_local+ threadsPerBlock - 1) / threadsPerBlock;

        int offset0=0;  // GPU 0 handles elements 0..n_elem_local-1

        assemble<<<blocksPerGrid, threadsPerBlock>>>(n_elem_local,n_elem_total, dx, dM_global0, dL_global0, dF_global0,offset0);


        cudaError_t err0 = cudaGetLastError();
        printf("GPU0 assemble kernel: %s\n", cudaGetErrorString(err0));
        


        // (2) for GPU1:
        double* dM_global1, *dL_global1, *dF_global1;
        cudaSetDevice(1);





        // err = cudaMalloc(&dM_global1, n_nodes_global * n_nodes_global * sizeof(double));
        // printf("GPU1 dM malloc: %s\n", cudaGetErrorString(err));
        // err = cudaMalloc(&dL_global1, n_nodes_global * n_nodes_global * sizeof(double));
        // printf("GPU1 dL malloc: %s\n", cudaGetErrorString(err));
        // err = cudaMalloc(&dF_global1, n_nodes_global * sizeof(double));
        // printf("GPU1 dF malloc: %s\n", cudaGetErrorString(err));


        cudaMalloc(&dM_global1, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dL_global1, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dF_global1, n_nodes_global * sizeof(double));

        cudaMemset(dM_global1, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dL_global1, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dF_global1, 0, n_nodes_global     * sizeof(double));

        int offset1 = n_elem_local;  // GPU 1 handles elements n_elem_local..n_elem_total-1 
        assemble<<<blocksPerGrid, threadsPerBlock>>>(n_elem_local,n_elem_total, dx, dM_global1, dL_global1, dF_global1,offset1);

        cudaError_t err1 = cudaGetLastError();
        printf("GPU1 assemble kernel: %s\n", cudaGetErrorString(err1));
        


        // NOW wait for both
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        cudaSetDevice(1);
        cudaDeviceSynchronize();


        ////////////////////////////// boundary halo exchange////////////////////////////////////////////////////////////////////////
        // ── Step 1: copy boundary region from both GPUs to CPU ──

                    // only need the boundary node row/col
                    // for simplicity copy full F vectors
                    std::vector<double> h_F0(n_nodes_global);
                    std::vector<double> h_F1(n_nodes_global);

                    cudaSetDevice(0);
                    cudaMemcpy(h_F0.data(), dF_global0,
                            n_nodes_global*sizeof(double),
                            cudaMemcpyDeviceToHost);

                    cudaSetDevice(1);
                    cudaMemcpy(h_F1.data(), dF_global1,
                            n_nodes_global*sizeof(double),
                            cudaMemcpyDeviceToHost);

                    // ── Step 2: add boundary node contribution ──────────────
                    // only node 5000 needs correction
                    // (only node shared between both GPUs)
                    int boundary_node = n_elem_local;  // = n_elem_total/2 = 5000

                    h_F0[boundary_node] += h_F1[boundary_node];

                    // ── Step 3: same for M and L matrices ───────────────────
                    // only the entries involving node 5000 need correction
                    // which are: (5000,5000), (5000,4999), (4999,5000)
                    // (5000,5001) and (5001,5000) are only on GPU 1 — no correction needed

                    std::vector<double> h_M0(n_nodes_global * n_nodes_global);
                    std::vector<double> h_M1(n_nodes_global * n_nodes_global);
                    std::vector<double> h_L0(n_nodes_global * n_nodes_global);
                    std::vector<double> h_L1(n_nodes_global * n_nodes_global);

                    cudaSetDevice(0);
                    cudaMemcpy(h_M0.data(), dM_global0,
                            n_nodes_global*n_nodes_global*sizeof(double),
                            cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_L0.data(), dL_global0,
                            n_nodes_global*n_nodes_global*sizeof(double),
                            cudaMemcpyDeviceToHost);

                    cudaSetDevice(1);
                    cudaMemcpy(h_M1.data(), dM_global1,
                            n_nodes_global*n_nodes_global*sizeof(double),
                            cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_L1.data(), dL_global1,
                            n_nodes_global*n_nodes_global*sizeof(double),
                            cudaMemcpyDeviceToHost);

                    // add boundary contributions
                    int b = boundary_node;
                    h_M0[idx(b, b, n_nodes_global)] += h_M1[idx(b, b, n_nodes_global)];
                    h_L0[idx(b, b, n_nodes_global)] += h_L1[idx(b, b, n_nodes_global)];

                    



        // add the GPU1 contribution to GPU0's global matrices and RHS, so GPU0 now has the full system assembled and can proceed to solve it.

        // for (int i = n_elem_local; i < n_nodes_global; i++) {
        //     for (int j = n_elem_local; j < n_nodes_global; j++) {
        //         h_M0[idx(i, j, n_nodes_global)] = h_M1[idx(i, j, n_nodes_global)];
        //         h_L0[idx(i, j, n_nodes_global)] = h_L1[idx(i, j, n_nodes_global)];
        //     }
        // }


        // for (int i = n_elem_local+1; i < n_nodes_global; i++) {
        //     h_F0[i] = h_F1[i];
        // }


        // ── Step 1: copy off-diagonal entries of row b from GPU 1 ──
            // (j > b only, because h_M0[b,b] was already summed above)
            for (int j = n_elem_local+1; j < n_nodes_global; j++) {
                h_M0[idx(b, j, n_nodes_global)] = h_M1[idx(b, j, n_nodes_global)];
                h_L0[idx(b, j, n_nodes_global)] = h_L1[idx(b, j, n_nodes_global)];
            }

            // ── Step 2: copy all rows after b entirely from GPU 1 ──
            for (int i = n_elem_local+1; i < n_nodes_global; i++) {
                for (int j = n_elem_local; j < n_nodes_global; j++) {
                    h_M0[idx(i, j, n_nodes_global)] = h_M1[idx(i, j, n_nodes_global)];
                    h_L0[idx(i, j, n_nodes_global)] = h_L1[idx(i, j, n_nodes_global)];
                }
            }



        





        // after merge loop
            printf("h_M1[b,b+1]   = %.6f\n", h_M1[idx(b, b+1, n_nodes_global)]);
            printf("h_M1[b+1,b+1] = %.6f\n", h_M1[idx(b+1, b+1, n_nodes_global)]);
            printf("h_M0[b,b+1] after merge = %.6f\n", h_M0[idx(b, b+1, n_nodes_global)]);
            printf("h_M0[b+1,b+1] after merge = %.6f\n", h_M0[idx(b+1, b+1, n_nodes_global)]);




        // debug for the NAN

        // check a few values around the boundary node
        printf("h_M0[b,b]   = %.6f\n", h_M0[idx(b, b, n_nodes_global)]);
        printf("h_M0[b-1,b] = %.6f\n", h_M0[idx(b-1, b, n_nodes_global)]);
        printf("h_M0[b,b+1] = %.6f\n", h_M0[idx(b, b+1, n_nodes_global)]);
        printf("h_F0[b]     = %.6f\n", h_F0[b]);
        printf("h_F0[b+1]   = %.6f\n", h_F0[b+1]);





































        // apply the boundary condition on the full system 
        apply_dirichlet(n_elem_total, h_M0, h_L0, h_F0);




        // extract diagonals directly on CPU — no need to copy back to GPU
std::vector<double> dl(n_elem_total-1), d(n_elem_total), du(n_elem_total-1), F(n_elem_total);

for (int i = 0; i < n_elem_total; i++) {
    d[i]  = h_M0[idx(i, i, n_elem_total)] + h_L0[idx(i, i, n_elem_total)];
    if (i > 0)
        dl[i-1] = h_M0[idx(i, i-1, n_elem_total)] + h_L0[idx(i, i-1, n_elem_total)];
    if (i < n_elem_total-1)
        du[i]   = h_M0[idx(i, i+1, n_elem_total)] + h_L0[idx(i, i+1, n_elem_total)];
    F[i] = h_F0[i];
}


        // // cpoy back h_M0, h_L0, h_F0 to GPU 0:
        // cudaSetDevice(0);
        // cudaMemcpy(dM_global0, h_M0.data(),
        //         n_nodes_global*n_nodes_global*sizeof(double),
        //         cudaMemcpyHostToDevice);
        // cudaMemcpy(dL_global0, h_L0.data(),
        //         n_nodes_global*n_nodes_global*sizeof(double),
        //         cudaMemcpyHostToDevice);
        // cudaMemcpy(dF_global0, h_F0.data(),
        //         n_nodes_global*sizeof(double),
        //         cudaMemcpyHostToDevice);


        // // now I have: dM_global0, dL_global0, dF_global0 with the full system assembled on GPU 0, I can proceed to solve it using cuSparse or cuSolver as before.

        // // get dl, d, du from M_global+L_global for the tridiagonal solver:
        //   std::vector<double> dl(n_elem_total-1), d(n_elem_total), du(n_elem_total-1), F(n_elem_total);




        //    double *d_diag, *d_lower, *d_upper, *d_F_inner;
        //     cudaMalloc(&d_diag,    n_elem_total     * sizeof(double));
        //     cudaMalloc(&d_lower,  (n_elem_total-1)  * sizeof(double));
        //     cudaMalloc(&d_upper,  (n_elem_total-1)  * sizeof(double));
        //     cudaMalloc(&d_F_inner, n_elem_total     * sizeof(double));



        //      int blocksPerGrid_total = (n_elem_total+ threadsPerBlock - 1) / threadsPerBlock;


        //     extract_diagonals_and_F<<<blocksPerGrid_total, threadsPerBlock>>>(
        //     n_elem_total, dM_global0, dL_global0, dF_global0,
        //     d_diag, d_lower, d_upper, d_F_inner);
        //     cudaDeviceSynchronize();

        //     // only copy 3 small arrays (~0.24 MB for n=10000)
        //     cudaMemcpy(d.data(),  d_diag,    n_elem_total*sizeof(double),     cudaMemcpyDeviceToHost);
        //     cudaMemcpy(dl.data(), d_lower,   (n_elem_total-1)*sizeof(double), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(du.data(), d_upper,   (n_elem_total-1)*sizeof(double), cudaMemcpyDeviceToHost);
        //     cudaMemcpy(F.data(),  d_F_inner, n_elem_total*sizeof(double),     cudaMemcpyDeviceToHost);


       
        std::vector<double> u_inner=solveGPU_sparse(n_elem_total,dl, d, du,F, sparseHandle);




        // check first and last few values
        printf("u_inner[0]    = %.6f\n", u_inner[0]);
        printf("u_inner[1]    = %.6f\n", u_inner[1]);
        printf("u_inner[n-1]  = %.6f\n", u_inner[n_elem_total-1]);

        // check diagonals
        printf("d[0]    = %.6f\n", d[0]);
        printf("d[5000] = %.6f\n", d[5000]);
        printf("d[9999] = %.6f\n", d[9999]);
        printf("dl[0]   = %.6f\n", dl[0]);
        printf("du[0]   = %.6f\n", du[0]);


        auto t2 = high_resolution_clock::now();
        double t = duration<double>(t2-t1).count();
        f2 << n_elem << "\t" << t << "\n";


        // restore full solution with u(0)=1
        std::vector<double> u(n_nodes_global);
        u[0] = 1.0;
        for (int i = 0; i < n_elem_total; i++)
            u[i+1] = u_inner[i];
        


        // output u to check:

            ofstream file110("uFluid.txt",ios::app);

            for(int i=0;i<n_nodes_global;i++){
               

                    file110  << u[i] <<endl;
            }

        // errors
        double L2, Linf;
        compute_errors(u, x, L2, Linf);

        std::cout << n_elem << "\t" << L2 << "\t\t" << Linf << "\n";

       cudaSetDevice(0);

        cudaFree(d_diag);
        cudaFree(d_lower);
        cudaFree(d_upper);
        cudaFree(d_F_inner);


        cudaFree(dM_global0);
        cudaFree(dL_global0);
        cudaFree(dF_global0);

        cudaSetDevice(1);
        cudaFree(dM_global1);
        cudaFree(dL_global1);
        cudaFree(dF_global1);
        
    }
    cusolverDnDestroy(handle);
    cusparseDestroy(sparseHandle);  // cleanup cuSparse handle

    return 0; 

        
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




















       







        


        
      
}
