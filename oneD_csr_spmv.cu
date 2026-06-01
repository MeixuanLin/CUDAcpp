
// June 1st, 2026: this is the code for 1D CSR for sparse matrix-vector multiplication (SpMV).


#include <cuda_runtime.h>
#include <vector>
using namespace std;




// the kernel function:
__global__ void spmv(
    const float* values,
    const int*   col_idx,
    const int*   row_ptr,
    const float* x,
    float*       y,
    int M)
{
    // which warp am i?

    // these replaces the one thread per element:

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    // which thread am i within the warp? (0 to 31)
    int lane = threadIdx.x % 32;

    if (warpId >= M) return;

    

    // this warp's row starts and ends at:
    int row_start = row_ptr[warpId];
    int row_end   = row_ptr[warpId + 1];

    // (A) each thread accumulates a partial sum
    float val = 0.0f;
    for (int k = row_start + lane; k < row_end; k += 32) {
        val += values[k] * x[col_idx[k]];
    }

    // (B) reduction — add up all 32 partial sums
    for(int offset = 16; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if(lane == 0) {
        y[warpId] = val;
    }
}
    


// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {


    // (1) get h_A:
    vector<float> h_A(M * N);
    fill(h_A.begin(),h_A.end(),0.0);

    cudaMemcpy(h_A.data(), A, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // (2) Allocate 3 CPU arrays for CSR:
    //    h_values   (length nnz) h_col_idx  (length nnz)   h_row_ptr  (length M+1)

    vector<float> h_values(nnz);
    vector<int>   h_col_idx(nnz);
    vector<int>   h_row_ptr(M + 1);


    // (3) fill in the 3 CPU vectors:
    int k = 0;
    for (int i = 0; i < M; i++) {
        h_row_ptr[i] = k;
        for (int j = 0; j < N; j++) {
            if(h_A[i*N+j]!=0){
                h_values[k]  = h_A[i*N+j];
                h_col_idx[k] = j;
                k++;
            }
            
        }
    }
    h_row_ptr[M] = k;

    // (4) and (5)
    float* d_values;
    int*   d_col_idx;
    int*   d_row_ptr;

    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_row_ptr, (M+1) * sizeof(int));

    // copy from CPU to GPU (extract is happening on CPU):
    cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (M+1) * sizeof(int), cudaMemcpyHostToDevice);


    // (6) launch the kernel:
    int threadsPerBlock = 256;   // 256 threads per block = 8 warps per block
    int blocks = (M + 7) / 8;   // ML: round up!

    // spmv(
    // const float* values,
    // const int*   col_idx,
    // const int*   row_ptr,
    // const float* x,
    // float*       y,
    // int M)
    // should input the GPU (device pointer)
    spmv<<<blocks, threadsPerBlock>>>(d_values, d_col_idx, d_row_ptr, x, y, M);

    cudaFree(d_values);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);





}
