#include "utils_1d.h"
#include <limits.h>

/* Implement the MPI solver here. You may add includes and helper functions.
 * Keep the explicit finite difference scheme and the public signature.
 * MPI is initialized/finalized by the driver. The full result and cumulative
 * error must be available on rank 0 when this function returns.
 */

 /**
 * @brief Solve the heat equation using an explicit finite difference scheme.
 *
 * This functions solves the heat equation iterating over ITER time steps.
 *
 * @param heat_array Array of `double` values representing the temperatures of the points.
 * @param err Pointer to a `double` value representing the error in the simulation.
 * @param num_pts Number of points in the simulation.
 *
 * @return 0 on success, -1 on error.
 */
int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts){
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (heat_array == NULL || err == NULL || num_pts == 0 ||
      num_pts > (size_t)INT_MAX) {
    return -1;
  }

  // Setup the parameters.  The boundary values are fixed at zero.
  double dx = 1.0/(num_pts+1);
  const double half_alpha = 0.5;

  int *counts = malloc((size_t)size * sizeof(*counts));
  int *displs = malloc((size_t)size * sizeof(*displs));
  if (counts == NULL || displs == NULL) {
    free(counts);
    free(displs);
    return -1;
  }

  size_t base = num_pts / (size_t)size;
  size_t remainder = num_pts % (size_t)size;
  size_t offset = 0;
  for (int r = 0; r < size; ++r) {
    size_t n = base + ((size_t)r < remainder ? 1 : 0);
    if (n > (size_t)INT_MAX || offset > (size_t)INT_MAX) {
      free(counts);
      free(displs);
      return -1;
    }
    counts[r] = (int)n;
    displs[r] = (int)offset;
    offset += n;
  }

  int local_n = counts[rank];
  double *current = malloc((size_t)(local_n + 2) * sizeof(*current));
  double *next = malloc((size_t)(local_n + 2) * sizeof(*next));
  if (current == NULL || next == NULL) {
    free(current);
    free(next);
    free(counts);
    free(displs);
    return -1;
  }

  MPI_Scatterv(heat_array + 1, counts, displs, MPI_DOUBLE,
               current + 1, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  current[0] = 0.0;
  current[local_n + 1] = 0.0;
  *err = 0.0;

  double local_err = 0.0;
  for (size_t step = 0; step < ITER; ++step) {
    int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Request requests[4];
    MPI_Irecv(&current[0], 1, MPI_DOUBLE, left, 1,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&current[local_n + 1], 1, MPI_DOUBLE, right, 0,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&current[1], 1, MPI_DOUBLE, left, 0,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(&current[local_n], 1, MPI_DOUBLE, right, 1,
              MPI_COMM_WORLD, &requests[3]);

    /* The inner points do not depend on the halo values. */
    const double *left_ptr = current + 1;
    const double *right_ptr = current + 3;
    double *next_ptr = next + 2;
    for (int j = 2; j < local_n; ++j) {
      *next_ptr = half_alpha * (*left_ptr + *right_ptr);
      local_err += dx * fabs(*next_ptr - current[j]);
      ++left_ptr;
      ++right_ptr;
      ++next_ptr;
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    if (local_n >= 1) {
      next[1] = half_alpha * (current[0] + current[2]);
      local_err += dx * fabs(next[1] - current[1]);
    }
    if (local_n >= 2) {
      next[local_n] = half_alpha *
          (current[local_n - 1] + current[local_n + 1]);
      local_err += dx * fabs(next[local_n] - current[local_n]);
    }

    double *tmp = current;
    current = next;
    next = tmp;
  }

  MPI_Gatherv(current + 1, local_n, MPI_DOUBLE,
              heat_array + 1, counts, displs, MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  MPI_Reduce(&local_err, err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  free(current);
  free(next);
  free(counts);
  free(displs);
  return 0;
}
