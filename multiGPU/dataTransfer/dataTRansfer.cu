// this is a demo in multi-GPu to teachme how to transfer data between different GPUs.

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>

#include <chrono>


using namespace std::chrono;

using namespace std;

int main(){

    // set device to GPU 0
        cudaSetDevice(0);



    // allocate memory on GPU 0
    int n=1000;
    double* d_data_0;
    cudaMalloc(&d_data_0, n*sizeof(double));


    // declare a host array and fill it with data
    vector<double> h_data(n);

    for(int i=0;i<n;i++){
        h_data[i] = i*1.0;
    }

    // on GPU 0: cpopy the date from host to device:
    cudaMemcpy(d_data_0,h_data.data(), n*sizeof(double), cudaMemcpyHostToDevice);


    // set device to GPU 1

    cudaSetDevice(1);

   // allocate memory on GPU 1
    double* d_data_1;
    cudaMalloc(&d_data_1, n*sizeof(double));

    // copy from CPU: h_data -> d_data_1
    
    vector<double> h_data_tem(n);

    // Pattern A timing
    auto t1 = high_resolution_clock::now();

    cudaMemcpy(h_data_tem.data(), d_data_0, n*sizeof(double), cudaMemcpyDeviceToHost);

    // copy from CPU: h_data_tem -> d_data_1
    cudaMemcpy(d_data_1, h_data_tem.data(), n*sizeof(double), cudaMemcpyHostToDevice);


    auto t2 = high_resolution_clock::now();
    double t_A = duration<double>(t2-t1).count() * 1000;
    printf("Pattern A time: %.4f ms\n", t_A);



    // Pattern C timing

    t1 = high_resolution_clock::now();
    cudaMemcpyPeer(d_data_1, 1, d_data_0, 0, n*sizeof(double));
    t2 = high_resolution_clock::now();
    double t_C = duration<double>(t2-t1).count() * 1000;
    printf("Pattern C time: %.4f ms\n", t_C);


    // check the results:

    vector<double> h_data_check(n);
    cudaMemcpy(h_data_check.data(), d_data_1, n*sizeof(double), cudaMemcpyDeviceToHost);

    // output h_data_check and h_data to txt file:

    ofstream file("data_check.txt",ios::out);
    for(int i=0;i<n;i++){
        file << h_data_check[i] << " " << h_data[i] << endl;
    }







    return 0;
}
