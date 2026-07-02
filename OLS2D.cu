#include <cuda_runtime.h>

// Kernel definition — just like a normal function with __global__
__global__ void compute_XtX(const float* X, float* A, int n_samples, int n_features) {


    //// step 1: define the row and column index i and j:

    // row index
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    // column index
    int j = blockIdx.x*blockDim.x+threadIdx.x;

    //// step 2: bound check:
    if (i >= n_features || j >= n_features) return;

        /// step 3: dot product
       

        // GOOD — n_samples fast register accumulations, one global write
        float sum = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            sum += X[s * n_features + i] * X[s * n_features + j];
        }
        A[i * n_features + j] = sum;
}

__global__ void compute_XtY(const float* X, const float* y, float* b, int n_samples, int n_features) {
    // b is a vector of length n_features
    // each thread computes one entry b[i]

    // Step 1: index (1D, so use x only)
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // here i is column

    // Step 2: bounds check
    if (i >= n_features) return;

    // Step 3: dot product — column i of X dotted with y
    float sum = 0.0f;
    for (int s = 0; s < n_samples; s++) {
        sum += X[s * n_features + i] * y[s];
    }

    // Step 4: write result
    b[i] = sum;
}


// kernel 3: solve the linear system:


// X, y, beta are device pointers
// Launch site — this is where <<<>>> goes
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    int nf = n_features;
    int ns = n_samples;

    // =============================================
    // Allocate GPU memory for A and b
    // =============================================
    float* A;
    float* b;
    cudaMalloc(&A, nf * nf * sizeof(float));
    cudaMalloc(&b, nf * sizeof(float));

    // =============================================
    // Step 1: Compute A = X^T X on GPU
    // =============================================
    dim3 threads2D(16, 16);
    dim3 blocks2D((nf + 15) / 16, (nf + 15) / 16);
    compute_XtX<<<blocks2D, threads2D>>>(X, A, ns, nf);

    // =============================================
    // Step 2: Compute b = X^T y on GPU
    // =============================================
    int threads1D = 256;
    int blocks1D = (nf + 255) / 256;
    compute_XtY<<<blocks1D, threads1D>>>(X, y, b, ns, nf);

    // =============================================
    // Step 3: Copy A and b back to CPU
    // =============================================
    cudaDeviceSynchronize();  // wait for both kernels to finish

    float* h_A = (float*)malloc(nf * nf * sizeof(float));
    float* h_b = (float*)malloc(nf * sizeof(float));
    float* h_beta = (float*)malloc(nf * sizeof(float));

    cudaMemcpy(h_A, A, nf * nf * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, b, nf * sizeof(float), cudaMemcpyDeviceToHost);

    // =============================================
    // Step 4: Gaussian elimination with partial pivoting (CPU)
    // =============================================

    // Forward elimination
    for (int k = 0; k < nf; k++) {
        // Partial pivoting — find the row with largest value in column k
        int max_row = k;
        float max_val = fabsf(h_A[k * nf + k]);
        for (int p = k + 1; p < nf; p++) {
            float val = fabsf(h_A[p * nf + k]);
            if (val > max_val) {
                max_val = val;
                max_row = p;
            }
        }

        // Swap rows k and max_row (in both A and b)
        if (max_row != k) {
            for (int c = 0; c < nf; c++) {
                float tmp = h_A[k * nf + c];
                h_A[k * nf + c] = h_A[max_row * nf + c];
                h_A[max_row * nf + c] = tmp;
            }
            float tmp = h_b[k];
            h_b[k] = h_b[max_row];
            h_b[max_row] = tmp;
        }

        // Eliminate column k in all rows below
        for (int row = k + 1; row < nf; row++) {
            float factor = h_A[row * nf + k] / h_A[k * nf + k];
            for (int c = k; c < nf; c++) {
                h_A[row * nf + c] -= factor * h_A[k * nf + c];
            }
            h_b[row] -= factor * h_b[k];
        }
    }

    // Back-substitution
    for (int i = nf - 1; i >= 0; i--) {
        h_beta[i] = h_b[i];
        for (int j = i + 1; j < nf; j++) {
            h_beta[i] -= h_A[i * nf + j] * h_beta[j];
        }
        h_beta[i] /= h_A[i * nf + i];
    }

    // =============================================
    // Step 5: Copy beta back to GPU
    // =============================================
    cudaMemcpy(beta, h_beta, nf * sizeof(float), cudaMemcpyHostToDevice);

    // =============================================
    // Free all memory
    // =============================================
    free(h_A);
    free(h_b);
    free(h_beta);
    cudaFree(A);
    cudaFree(b);
}