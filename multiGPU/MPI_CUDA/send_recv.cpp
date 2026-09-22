// this is the code to teach me how to use MPI+CUDA to do send/recv between two GPUs on different processes
// 0526: Tuesday

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double data = 0.0;

    if (rank == 0) {
        // rank 0 sends to rank 1
        data = 42.0;
        MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);  // 1: destanation rank
        printf("rank 0 sent: %.0f\n", data);
    }
    if (rank == 1) {
        // rank 1 receives from rank 0
        MPI_Recv(&data, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 1 received: %.0f\n", data);
    }

    MPI_Finalize();
    return 0;
}