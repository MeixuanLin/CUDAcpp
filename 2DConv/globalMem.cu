#include <cuda_runtime.h>


//(1) this is a global memory version

__global__ void convolution_2d_kernel(const float* input,const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols){


                        // step 0: define the output row and cols

                        int output_rows = input_rows - kernel_rows + 1;
                        int output_cols = input_cols - kernel_cols + 1;

                        // step 1: define the index and the threads id:

                        int row=blockIdx.x*blockDim.x+threadIdx.x;
                        int col=blockIdx.y*blockDim.y+threadIdx.y;

                        // step 2: check the outbound:

                        if(row >= output_rows|| col >= output_cols) return;



                        // step 3: real computation:
                        float sum=0.0f;
                       for(int m=0;m<kernel_rows;m++){
                        for(int n=0;n<kernel_cols;n++){

                          
                          sum+=input[(row+m)*input_cols+(col+n)]*kernel[m*kernel_cols+n];
                         

                        }
                       }
                        
                         output[row*output_cols+col]=sum;




}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {

                        int output_rows = input_rows - kernel_rows + 1;
                        int output_cols = input_cols - kernel_cols + 1;

                        dim3 threadsPerBlock(16,16); 

                        
                        dim3 blocksPerGrid((output_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

                       //matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
                        convolution_2d_kernel<<< blocksPerGrid,threadsPerBlock >>>(input, kernel,  output,  input_rows,
                      input_cols,  kernel_rows,  kernel_cols);


                       





                       

                        cudaDeviceSynchronize();



                      }
