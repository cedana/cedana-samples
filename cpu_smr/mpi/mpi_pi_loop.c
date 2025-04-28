#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // Required for sleep()

int main(int argc, char *argv[]) {
  int rank, size, i, error_code;
  long long n; // Number of intervals
  double mypi, pi, h, sum, x;
  double iter_start_time = 0.0, iter_end_time = 0.0;
  long long loop_count = 0; // Iteration counter

  // --- Standard MPI Initialization ---
  error_code = MPI_Init(&argc, &argv);
  if (error_code != MPI_SUCCESS) {
    fprintf(stderr, "Error initializing MPI!\n");
    MPI_Abort(MPI_COMM_WORLD, error_code);
    exit(1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // --- Input Handling (only needed on root process) ---
  if (rank == 0) {
    if (argc != 2) {
      fprintf(stderr, "Usage: mpirun -np <num_procs> %s <num_intervals>\n",
              argv[0]);
      fprintf(stderr, "       (Will run indefinitely until Ctrl+C)\n");
      n = -1; // Signal error
    } else {
      n = atoll(argv[1]); // Convert argument to long long integer
      if (n <= 0) {
        fprintf(stderr, "Error: Number of intervals must be positive.\n");
        n = -1; // Signal error
      }
    }
  }

  // --- Broadcast the number of intervals (n) ---
  MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  // If input was invalid on root, all processes should exit.
  if (n <= 0) {
    MPI_Finalize();
    exit(1);
  }

  // --- Main Calculation Loop (Infinite) ---
  while (1) { // Loop forever
    if (rank == 0) {
      loop_count++;
      iter_start_time = MPI_Wtime(); // Time each iteration
      printf("--- Iteration %lld ---\n", loop_count);
    }

    // Broadcast start time measurement (optional, for more sync if needed)
    // MPI_Bcast(&iter_start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    h = 1.0 / (double)n;
    sum = 0.0; // Reset sum for each iteration

    long long start_index = rank * (n / size);
    long long end_index = (rank + 1) * (n / size);
    if (rank == size - 1) {
      end_index = n;
    }

    for (i = start_index; i < end_index; i++) {
      x = h * ((double)i + 0.5);
      sum += (4.0 / (1.0 + x * x));
    }
    mypi = h * sum;

    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      iter_end_time = MPI_Wtime();
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
      printf("Iter %lld: Calculated pi = %.16f (Error=%.3e) Time=%.4f s\n",
             loop_count, pi, fabs(pi - M_PI), iter_end_time - iter_start_time);
      sleep(1); // Sleep for 1 second
    }
  }
  MPI_Finalize();
  return 0;
}
