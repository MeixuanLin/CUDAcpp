#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // each rank has its own local array
    int n_local = 3;
    double local_data[3];
    for (int i = 0; i < n_local; i++)
        local_data[i] = rank * 10.0 + i;  

    // rank 0: [0, 1, 2]
    // rank 1: [10, 11, 12]

    // rank 0 collects everything into this buffer
    double global_data[6] = {0.0};  // size = n_local * size = 3 * 2

    MPI_Gather(
        local_data,    // what each rank sends
        n_local,       // how many elements each rank sends
        MPI_DOUBLE,    // data type
        global_data,   // where rank 0 collects (only meaningful on rank 0)
        n_local,       // how many elements to receive from each rank
        MPI_DOUBLE,    // data type
        0,             // root rank (who collects)
        MPI_COMM_WORLD
    );

    // only rank 0 has the full result
    if (rank == 0) {
        printf("global_data: ");
        for (int i = 0; i < 6; i++)
            printf("%.0f ", global_data[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}