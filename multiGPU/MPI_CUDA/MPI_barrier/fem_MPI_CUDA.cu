//  1D Helmholtz FEM: -u'' + u = f,  u(0)=1,  u'(1)=0
//  Exact solution: u = cos(2*pi*x)
// 0514: multi-GPU version


//0523 2026: this is the code for MPI+CUDA

// 0527: MPI+CUDA: add the M[I into the code:


// 0528: this is for adding the mpi_barrier for timing the wholr


#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <fstream>
#include <cusparse.h>
#include <cusolverDn.h>
#include <chrono>


#include <mpi.h>


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
        int gi = conn[i];  // i=0: 1st node of the element, i=1: 2nd node of the element
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
{  // 0703: cuSparse requires all the array of the same length n, so padding
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
    

    // solve:

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



// MPI version — needs these
int main(int argc, char** argv)

{


    MPI_Init(&argc, &argv);

    // get the MPI rank and size

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);




    cout << "this is the multi-GPU code for 1D Helmholtz FEM" << endl;
    std::ofstream f2("timing_MPIGPU_MPIBarrier.txt");
    f2 << "n\ttime\n";

    

    cudaSetDevice(rank);  // set once, never change

    // warmup
    
    // double *d_dummy; cudaMalloc(&d_dummy, sizeof(double)); cudaFree(d_dummy);

    // // GPU test
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

  
    printf("n\tL2 error\t\tLinf error\n");
    printf("------------------------------------------------\n");

    std::vector<int> n_list = {500, 1000, 5000};

    for (int n_elem : n_list) {


         int n_elem_local   = n_elem / size;       // ← uses n_elem
        int n_nodes_global = n_elem + 1;          // ← uses n_elem
        double dx          = 1.0 / n_elem;





      


        // node coordinates
        std::vector<double> x(n_nodes_global);
        for (int i = 0; i < n_nodes_global; i++)
            x[i] = i * dx;


        MPI_Barrier(MPI_COMM_WORLD);  // synchronize before timing

        auto t1 = high_resolution_clock::now();

       



        /// step 5: in MPI: each rank done 1 assembly: (this equals to combine the two GPUs work into the rank)

        double *dM, *dL, *dF;

        cudaMalloc(&dM, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dL, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMalloc(&dF, n_nodes_global * sizeof(double));

        cudaMemset(dM, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dL, 0, n_nodes_global * n_nodes_global * sizeof(double));
        cudaMemset(dF, 0, n_nodes_global * sizeof(double));


        int threadsPerBlock = 256;
        int blocksPerGrid   = (n_elem_local + threadsPerBlock - 1) / threadsPerBlock;

        
        int offset=rank*n_elem_local;


    //     printf("rank %d: n_elem_local=%d, offset=%d, dx=%.6f, blocksPerGrid=%d, threadsPerBlock=%d\n",
    //    rank, n_elem_local, offset, dx, blocksPerGrid, threadsPerBlock);
        assemble<<<blocksPerGrid, threadsPerBlock>>>(n_elem_local, n_elem, dx, dM, dL, dF, offset);

        // cudaError_t err = cudaGetLastError();
        // printf("rank %d: kernel launch: %s\n", rank, cudaGetErrorString(err));

        cudaDeviceSynchronize();



        

        // debug: check GPU memory after assembly
        double h_test_val = 0.0;
        cudaMemcpy(&h_test_val, &dM[idx(0,0,n_nodes_global)], 
                sizeof(double), cudaMemcpyDeviceToHost);
        //printf("rank %d: dM[0,0] after assembly = %.10f\n", rank, h_test_val);

        cudaMemcpy(&h_test_val, &dF[0], 
                sizeof(double), cudaMemcpyDeviceToHost);
        //printf("rank %d: dF[0] after assembly = %.10f\n", rank, h_test_val);

        



        




        ///  step 6: each rank copy from GPU to CPU:


        std::vector<double> h_M(n_nodes_global * n_nodes_global);
        std::vector<double> h_L(n_nodes_global * n_nodes_global);
        std::vector<double> h_F(n_nodes_global);

        cudaMemcpy(h_M.data(), dM, n_nodes_global*n_nodes_global*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_L.data(), dL, n_nodes_global*n_nodes_global*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_F.data(), dF, n_nodes_global*sizeof(double),                cudaMemcpyDeviceToHost);



    
        // step 7: boundary correction with MPI:

        // for h(b,b):
        // h_M0[idx(b, b, n_nodes_global)] += h_M1[idx(b, b, n_nodes_global)];  // after this, h_M0 is the contribution from both elements
        // h_L0[idx(b, b, n_nodes_global)] += h_L1[idx(b, b, n_nodes_global)];
        // h_F0[b] += h_F1[b];

        int b            = n_elem_local;
        int n_send_row   = n_nodes_global - (b+1);
        int n_rows       = n_nodes_global - b - 1;
        int n_send_block = n_rows * n_nodes_global;
        int n_f          = n_nodes_global - b - 1;

        if (rank == 1) {
            // (7.1) scalars
            MPI_Send(&h_M[idx(b,b,n_nodes_global)],      1,            MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&h_L[idx(b,b,n_nodes_global)],      1,            MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&h_F[b],                            1,            MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
            // (7.2) row b, j > b
            MPI_Send(&h_M[idx(b,b+1,n_nodes_global)],    n_send_row,   MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
            MPI_Send(&h_L[idx(b,b+1,n_nodes_global)],    n_send_row,   MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
            // (7.3) lower block
            MPI_Send(&h_M[idx(b+1,0,n_nodes_global)],    n_send_block, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
            MPI_Send(&h_L[idx(b+1,0,n_nodes_global)],    n_send_block, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
            MPI_Send(&h_F[b+1],                          n_f,          MPI_DOUBLE, 0, 8, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            // (7.1) scalars
            double recv_M, recv_L, recv_F;
            MPI_Recv(&recv_M, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            h_M[idx(b,b,n_nodes_global)] += recv_M;
            MPI_Recv(&recv_L, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            h_L[idx(b,b,n_nodes_global)] += recv_L;
            MPI_Recv(&recv_F, 1, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            h_F[b] += recv_F;
            // (7.2) row b, j > b
            MPI_Recv(&h_M[idx(b,b+1,n_nodes_global)],    n_send_row,   MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&h_L[idx(b,b+1,n_nodes_global)],    n_send_row,   MPI_DOUBLE, 1, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // (7.3) lower block
            MPI_Recv(&h_M[idx(b+1,0,n_nodes_global)],    n_send_block, MPI_DOUBLE, 1, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&h_L[idx(b+1,0,n_nodes_global)],    n_send_block, MPI_DOUBLE, 1, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&h_F[b+1],                          n_f,          MPI_DOUBLE, 1, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }



        // 0528: debug: 
        // if (rank == 0) {
        //         printf("\n=== Debug rank 0 after boundary correction ===\n");
        //         printf("h_M[b,b]   = %.10f\n", h_M[idx(b,b,n_nodes_global)]);
        //         printf("h_M[b-1,b] = %.10f\n", h_M[idx(b-1,b,n_nodes_global)]);
        //         printf("h_M[b,b+1] = %.10f\n", h_M[idx(b,b+1,n_nodes_global)]);
        //         printf("h_F[0]     = %.10f\n", h_F[0]);
        //         printf("h_F[b]     = %.10f\n", h_F[b]);
        //         printf("h_F[b+1]   = %.10f\n", h_F[b+1]);
        //         printf("h_F[n-1]   = %.10f\n", h_F[n_nodes_global-1]);
        // }


        // step 8: now everything is in rank 0:

        if(rank==0){
        // ── apply Dirichlet BC ────────────────────────────────────
                apply_dirichlet(n_elem, h_M, h_L, h_F);

                // ── extract diagonals on CPU ──────────────────────────────
                std::vector<double> dl(n_elem-1), d(n_elem), du(n_elem-1), F(n_elem);
                for (int i = 0; i < n_elem; i++) {
                    d[i] = h_M[idx(i, i, n_elem)] + h_L[idx(i, i, n_elem)];
                    if (i > 0)
                        dl[i-1] = h_M[idx(i, i-1, n_elem)] + h_L[idx(i, i-1, n_elem)];
                    if (i < n_elem-1)
                        du[i]   = h_M[idx(i, i+1, n_elem)] + h_L[idx(i, i+1, n_elem)];
                    F[i] = h_F[i];
                }



               // ML : call the cuSparse tridiagonal solver to solve the system on GPU:
                std::vector<double> u_inner = solveGPU_sparse(n_elem, dl, d, du, F, sparseHandle);


                

                // ── restore full solution ─────────────────────────────────
                std::vector<double> u(n_nodes_global);
                u[0] = 1.0;
                for (int i = 0; i < n_elem; i++)
                    u[i+1] = u_inner[i];



                 
                // ── errors ────────────────────────────────────────────────
                double L2, Linf;
                compute_errors(u, x, L2, Linf);
                printf("%d\t%.6e\t\t%.6e\n", n_elem, L2, Linf);

            }

            MPI_Barrier(MPI_COMM_WORLD);  // synchronize before timing

                auto t2 = high_resolution_clock::now();


                if (rank == 0) {
                    double t = duration<double>(t2-t1).count();
                    f2 << n_elem << "\t" << t << "\n";
                    printf("Time for n=%d: %.4f seconds\n", n_elem, t);
                }

                
    


              cudaFree(dM); cudaFree(dL); cudaFree(dF);  // each rank frees its own
            






        
       

       
    } // close the for loop

     // ── cleanup ───────────────────────────────────────────────
     
        cusolverDnDestroy(handle);
        cusparseDestroy(sparseHandle);
        MPI_Finalize();  // must be last


    return 0;
}  // close of main