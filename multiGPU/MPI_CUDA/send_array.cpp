#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {


    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 5;
    double data[5] = {0.0};

    if (rank == 0) {
        // fill array
        for (int i = 0; i < n; i++)
            data[i] = i * 10.0;  // 0, 10, 20, 30, 40

        // this is for sending 1 number:

        //MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);  // 1: destanation rank

        MPI_Send(data, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        printf("rank 0 sent: %.0f %.0f %.0f %.0f %.0f\n",
               data[0], data[1], data[2], data[3], data[4]);
    }

    if (rank == 1) {
        MPI_Recv(data, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 1 received: %.0f %.0f %.0f %.0f %.0f\n",
               data[0], data[1], data[2], data[3], data[4]);
    }

    MPI_Finalize();
    return 0;
}