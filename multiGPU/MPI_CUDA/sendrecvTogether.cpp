#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double send_val = rank * 10.0;  // rank 0 sends 0.0, rank 1 sends 10.0
    double recv_val = 0.0;  // ML: empty/waiting, MPI fills it with whatever the other rank sends (can be also seen as a buffer)

    MPI_Sendrecv(
        &send_val, 1, MPI_DOUBLE, 1-rank, 0,   // send to the other rank
        &recv_val, 1, MPI_DOUBLE, 1-rank, 0,   // receive from the other rank
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    printf("rank %d sent %.0f, received %.0f\n", rank, send_val, recv_val);

    MPI_Finalize();
    return 0;
}