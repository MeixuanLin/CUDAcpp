#include <cuda_runtime.h>

__global__ void conv3D(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols){

                        // (i) define thread id (index)

                        int i=blockIdx.x*blockDim.x+threadIdx.x;
                        int j=blockIdx.y*blockDim.y+threadIdx.y;
                        int k=blockIdx.z*blockDim.z+threadIdx.z;

                        // define output size:
                        int output_depth = input_depth - kernel_depth + 1;
                        int output_rows = input_rows - kernel_rows + 1;
                        int output_cols = input_cols - kernel_cols + 1;


                        // (ii) check the out of bound:

                        if(i>=output_rows || j>=output_cols || k >=output_depth) return;


                        // (iii) the actual kernel computation:

                        float sum=0.0f;

                        // for(int a=0;a<kernel_rows;a++){
                        //     for(int b=0;b<kernel_cols;b++){
                        //         for(int c=0;c<kernel_depth;c++){
                        //             sum+=input[(i+a)*input_cols* input_depth+(j+b)*input_depth+(k+c)]*kernel[a*kernel_cols*kernel_depth+b*kernel_depth+c];
                        //         }
                        //     }
                        // }

                        for(int c=0;c<kernel_depth;c++){
                            for(int a=0;a<kernel_rows;a++){
                                for(int b=0;b<kernel_cols;b++){
                                    sum += input[(k+c)*input_rows*input_cols + (i+a)*input_cols + (j+b)]
                                        * kernel[c*kernel_rows*kernel_cols + a*kernel_cols + b];
                                }
                            }
                        }

                        output[i*output_cols*output_depth+j*output_depth+k]=sum;


}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {


    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 threadsPerBlock(8,8,4);  // 3D block
   dim3 blocksPerGrid(
    (output_rows  + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (output_cols  + threadsPerBlock.y - 1) / threadsPerBlock.y,
    (output_depth + threadsPerBlock.z - 1) / threadsPerBlock.z
);
    
    conv3D<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output,
             input_depth, input_rows, input_cols,
             kernel_depth, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
     }
