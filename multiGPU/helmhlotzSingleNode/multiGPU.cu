//  1D Helmholtz FEM: -u'' + u = f,  u(0)=1,  u'(1)=0
//  Exact solution: u = cos(2*pi*x)
// 0514: multi-GPU version

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <cusparse.h>
#include <cusolverDn.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

#define CUDA_CHECK(call) { cudaError_t err = call; if(err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }

const double PI = acos(-1.0);

__host__ __device__ inline int idx(int i, int j, int n) { return i * n + j; }

__device__ __constant__ double PI_d = 3.14159265358979323846;

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

__device__ void compute_element_matrices(double dx, double x_left,
                                          double Me[4], double Le[4], double fe[2])
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
            Me[i*2+j] = Je * (phi[i*2+0]*phi[j*2+0] + phi[i*2+1]*phi[j*2+1]);

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

__global__ void assemble(int n_elem_local, int n_elem_total, double dx,
                          double* dM_global, double* dL_global, double* dF_global,
                          int offset)
{
    int n_nodes = n_elem_total + 1;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem_local) return;

    int e_global = e + offset;
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

void apply_dirichlet(int n_elem,
                     std::vector<double>& M,
                     std::vector<double>& L,
                     std::vector<double>& F)
{
    int n = n_elem + 1;
    double u0 = 1.0;
    for (int i = 1; i < n; i++)
        F[i] -= (M[idx(i,0,n)] + L[idx(i,0,n)]) * u0;
    for (int i = 0; i < n_elem; i++) {
        F[i] = F[i+1];
        for (int j = 0; j < n_elem; j++) {
            M[idx(i,j,n_elem)] = M[idx(i+1,j+1,n)];
            L[idx(i,j,n_elem)] = L[idx(i+1,j+1,n)];
        }
    }
}

std::vector<double> solveGPU_sparse(int n,
                                    std::vector<double>& dl,
                                    std::vector<double>& d,
                                    std::vector<double>& du,
                                    std::vector<double>& F,
                                    cusparseHandle_t handle)
{
    std::vector<double> dl_pad(n, 0.0), du_pad(n, 0.0);
    std::copy(dl.begin(), dl.end(), dl_pad.begin() + 1);
    std::copy(du.begin(), du.end(), du_pad.begin());

    double *dDl, *dD, *dDu, *dF;
    cudaMalloc(&dDl, n * sizeof(double));
    cudaMalloc(&dD,  n * sizeof(double));
    cudaMalloc(&dDu, n * sizeof(double));
    cudaMalloc(&dF,  n * sizeof(double));

    cudaMemcpy(dDl, dl_pad.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dD,  d.data(),      n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dDu, du_pad.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dF,  F.data(),      n * sizeof(double), cudaMemcpyHostToDevice);

    size_t bufferSize;
    cusparseDgtsv2_bufferSizeExt(handle, n, 1, dDl, dD, dDu, dF, n, &bufferSize);
    void* dBuffer;
    cudaMalloc(&dBuffer, bufferSize);

    cusparseDgtsv2(handle, n, 1, dDl, dD, dDu, dF, n, dBuffer);

    std::vector<double> result(n);
    cudaMemcpy(result.data(), dF, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dDl); cudaFree(dD); cudaFree(dDu);
    cudaFree(dF);  cudaFree(dBuffer);
    return result;
}

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

int main()
{




    cout << "this is the multi-GPU code for 1D Helmholtz FEM" << endl;
    std::ofstream f2("timing_multiGPU.txt");
    f2 << "n\ttime\n";

    int n_elem_total  = 1000;
    
    // warmup
    cudaSetDevice(0);
    double *d_dummy; cudaMalloc(&d_dummy, sizeof(double)); cudaFree(d_dummy);

    // GPU test
    // double test_val = 42.0;
    // double *d_test, h_test = 0.0;
    // CUDA_CHECK(cudaMalloc(&d_test, sizeof(double)));
    // CUDA_CHECK(cudaMemcpy(d_test, &test_val, sizeof(double), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(&h_test, d_test, sizeof(double), cudaMemcpyDeviceToHost));
    // printf("GPU test: sent 42, got %.0f\n", h_test);
    // cudaFree(d_test);

    // handles
    cusolverDnHandle_t handle; cusolverDnCreate(&handle);
    cusparseHandle_t sparseHandle; cusparseCreate(&sparseHandle);

    std::vector<int> n_list = {n_elem_total/2, n_elem_total,n_elem_total*5};  // test both halves
    printf("n\tL2 error\t\tLinf error\n");
    printf("------------------------------------------------\n");

    for (int n_elem : n_list) {

        int n_elem_local  = n_elem / 2;
        int n_nodes_global = n_elem + 1;


        double dx = 1.0 / n_elem;

        // node coordinates
        std::vector<double> x(n_nodes_global);
        for (int i = 0; i < n_nodes_global; i++)
            x[i] = i * dx;

        auto t1 = high_resolution_clock::now();

        // ── GPU 0 assembly ────────────────────────────────────────
        double *dM0, *dL0, *dF0;
        cudaSetDevice(0);
        cudaMalloc(&dM0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dL0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dF0, n_nodes_global * sizeof(double));
        cudaMemset(dM0, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dL0, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dF0, 0, n_nodes_global * sizeof(double));

        int threadsPerBlock = 256;
        int blocksPerGrid   = (n_elem_local + threadsPerBlock - 1) / threadsPerBlock;

        assemble<<<blocksPerGrid, threadsPerBlock>>>(n_elem_local, n_elem, dx, dM0, dL0, dF0, 0);

        // ── GPU 1 assembly ────────────────────────────────────────
        double *dM1, *dL1, *dF1;
        cudaSetDevice(1);
        cudaMalloc(&dM1, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dL1, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dF1, n_nodes_global * sizeof(double));
        cudaMemset(dM1, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dL1, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dF1, 0, n_nodes_global * sizeof(double));

        assemble<<<blocksPerGrid, threadsPerBlock>>>(n_elem_local, n_elem, dx, dM1, dL1, dF1, n_elem_local);

        // ── wait for both ─────────────────────────────────────────
        cudaSetDevice(0); cudaDeviceSynchronize();
        cudaSetDevice(1); cudaDeviceSynchronize();

        // ── copy both GPUs to CPU ─────────────────────────────────
        std::vector<double> h_M0(n_nodes_global * n_nodes_global);
        std::vector<double> h_L0(n_nodes_global * n_nodes_global);
        std::vector<double> h_F0(n_nodes_global);
        std::vector<double> h_M1(n_nodes_global * n_nodes_global);
        std::vector<double> h_L1(n_nodes_global * n_nodes_global);
        std::vector<double> h_F1(n_nodes_global);

        cudaSetDevice(0);
        cudaMemcpy(h_M0.data(), dM0, n_nodes_global*n_nodes_global*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_L0.data(), dL0, n_nodes_global*n_nodes_global*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_F0.data(), dF0, n_nodes_global*sizeof(double),                cudaMemcpyDeviceToHost);

        cudaSetDevice(1);
        cudaMemcpy(h_M1.data(), dM1, n_nodes_global*n_nodes_global*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_L1.data(), dL1, n_nodes_global*n_nodes_global*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_F1.data(), dF1, n_nodes_global*sizeof(double),                cudaMemcpyDeviceToHost);

        // ── boundary correction ───────────────────────────────────
        int b = n_elem_local;  // boundary node = 5000

        // sum boundary diagonal
        h_M0[idx(b, b, n_nodes_global)] += h_M1[idx(b, b, n_nodes_global)];
        h_L0[idx(b, b, n_nodes_global)] += h_L1[idx(b, b, n_nodes_global)];
        h_F0[b] += h_F1[b];

        // copy off-diagonal entries of row b from GPU 1 (j > b only)
        for (int j = b+1; j < n_nodes_global; j++) {
            h_M0[idx(b, j, n_nodes_global)] = h_M1[idx(b, j, n_nodes_global)];
            h_L0[idx(b, j, n_nodes_global)] = h_L1[idx(b, j, n_nodes_global)];
        }

        // copy all rows after b from GPU 1
        for (int i = b+1; i < n_nodes_global; i++) {
            for (int j = b; j < n_nodes_global; j++) {
                h_M0[idx(i, j, n_nodes_global)] = h_M1[idx(i, j, n_nodes_global)];
                h_L0[idx(i, j, n_nodes_global)] = h_L1[idx(i, j, n_nodes_global)];
            }
            h_F0[i] = h_F1[i];
        }

        // ── apply Dirichlet BC ────────────────────────────────────
        apply_dirichlet(n_elem, h_M0, h_L0, h_F0);

        // ── extract diagonals on CPU ──────────────────────────────
        std::vector<double> dl(n_elem-1), d(n_elem), du(n_elem-1), F(n_elem);
        for (int i = 0; i < n_elem; i++) {
            d[i] = h_M0[idx(i, i, n_elem)] + h_L0[idx(i, i, n_elem)];
            if (i > 0)
                dl[i-1] = h_M0[idx(i, i-1, n_elem)] + h_L0[idx(i, i-1, n_elem)];
            if (i < n_elem-1)
                du[i]   = h_M0[idx(i, i+1, n_elem)] + h_L0[idx(i, i+1, n_elem)];
            F[i] = h_F0[i];
        }



        // add after extract diagonals, before solve:
            // printf("\n=== Diagonal check ===\n");
            // printf("d[0]    = %.10f  (expected ~0.000134)\n", d[0]);
            // printf("d[1]    = %.10f  (expected ~0.000134)\n", d[1]);
            // printf("d[4999] = %.10f  (expected ~0.000134)\n", d[4999]);
            // printf("d[5000] = %.10f  (expected ~0.000134)\n", d[5000]);
            // printf("d[9999] = %.10f  (expected ~0.000134)\n", d[9999]);
            // printf("dl[0]   = %.10f  (expected ~-0.000067)\n", dl[0]);
            // printf("du[0]   = %.10f  (expected ~-0.000067)\n", du[0]);
            // printf("F[0]    = %.10f\n", F[0]);
            // printf("F[4999] = %.10f\n", F[4999]);
            // printf("F[5000] = %.10f\n", F[5000]);
            // printf("F[9999] = %.10f\n", F[9999]);

        // ── solve ─────────────────────────────────────────────────

        // make sure we are on GPU 0 before solving
        cudaSetDevice(0);
        std::vector<double> u_inner = solveGPU_sparse(n_elem, dl, d, du, F, sparseHandle);

        auto t2 = high_resolution_clock::now();
        double t = duration<double>(t2-t1).count();
        f2 << n_elem << "\t" << t << "\n";

        // ── restore full solution ─────────────────────────────────
        std::vector<double> u(n_nodes_global);
        u[0] = 1.0;
        for (int i = 0; i < n_elem; i++)
            u[i+1] = u_inner[i];

        // ── errors ────────────────────────────────────────────────
        double L2, Linf;
        compute_errors(u, x, L2, Linf);
        printf("%d\t%.6e\t\t%.6e\n", n_elem, L2, Linf);

        // ── cleanup ───────────────────────────────────────────────
        cudaSetDevice(0);
        cudaFree(dM0); cudaFree(dL0); cudaFree(dF0);
        cudaSetDevice(1);
        cudaFree(dM1); cudaFree(dL1); cudaFree(dF1);
    }

    cusolverDnDestroy(handle);
    cusparseDestroy(sparseHandle);
    return 0;
}