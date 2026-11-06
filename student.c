#include "utils_1d.h"
#include <limits.h>

int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (!heat_array || !err || !num_pts || num_pts > INT_MAX) return -1;

  const double dx = 1.0 / (num_pts + 1);
  *err = 0.0;
  if (num_pts < (size_t)size) {
    if (rank == 0) {
      double *next = malloc((num_pts + 2) * sizeof(double));
      if (!next) return -1;
      next[0] = next[num_pts + 1] = 0.0;
      double *a = heat_array, *b = next;
      for (int step = 0; step < ITER; ++step) {
        for (size_t j = 1; j <= num_pts; ++j) {
          b[j] = 0.5 * (a[j-1] + a[j+1]);
          *err += dx * fabs(b[j] - a[j]);
        }
        double *tmp = a; a = b; b = tmp;
      }
      if (a != heat_array) memcpy(heat_array + 1, a + 1, num_pts * sizeof(double));
      free(next);
    }
    return 0;
  }

  const int base = (int)(num_pts / (size_t)size);
  const int rem = (int)(num_pts % (size_t)size);
  const int n = base + (rank < rem);
  const int offset = rank * base + (rank < rem ? rank : rem);
  int width = base < 16 ? base : 16;
  const int own_lo = width + 1, own_hi = width + n;
  const int phys_lo = width - offset;
  const int phys_hi = width + (int)num_pts - offset + 1;
  double *a = malloc((size_t)(n + 2 * width + 2) * sizeof(double));
  double *b = malloc((size_t)(n + 2 * width + 2) * sizeof(double));
  int *counts = malloc((size_t)size * sizeof(int));
  int *displs = malloc((size_t)size * sizeof(int));
  if (!a || !b || !counts || !displs) {
    free(a); free(b); free(counts); free(displs);
    return -1;
  }
  for (int r = 0; r < size; ++r) {
    counts[r] = base + (r < rem);
    displs[r] = r * base + (r < rem ? r : rem);
  }
  memcpy(a + own_lo, heat_array + offset + 1, (size_t)n * sizeof(double));
  double local_err = 0.0;
  const int left = rank ? rank - 1 : MPI_PROC_NULL;
  const int right = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

  for (int step = 0; step < ITER; ) {
    int chunk = ITER - step < width ? ITER - step : width;
    MPI_Request req[4];
    MPI_Irecv(a + own_lo - width, width, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(a + own_hi + 1, width, MPI_DOUBLE, right, 1, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(a + own_lo, width, MPI_DOUBLE, left, 1, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(a + own_hi - width + 1, width, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (left == MPI_PROC_NULL) a[phys_lo] = 0.0;
    if (right == MPI_PROC_NULL) a[phys_hi] = 0.0;

    for (int t = 0; t < chunk; ++t) {
      int halo = chunk - t - 1;
      int lo = own_lo - halo;
      int hi = own_hi + halo;
      if (lo <= phys_lo) lo = phys_lo + 1;
      if (hi >= phys_hi) hi = phys_hi - 1;
      if (left == MPI_PROC_NULL) b[phys_lo] = 0.0;
      if (right == MPI_PROC_NULL) b[phys_hi] = 0.0;
      for (int j = lo; j < own_lo; ++j) b[j] = 0.5 * (a[j-1] + a[j+1]);
      for (int j = own_lo; j <= own_hi; ++j) {
        b[j] = 0.5 * (a[j-1] + a[j+1]);
        local_err += dx * fabs(b[j] - a[j]);
      }
      for (int j = own_hi + 1; j <= hi; ++j) b[j] = 0.5 * (a[j-1] + a[j+1]);
      double *tmp = a; a = b; b = tmp;
    }
    step += chunk;
  }
  MPI_Gatherv(a + own_lo, n, MPI_DOUBLE, heat_array + 1, counts, displs,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_err, err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  free(a); free(b); free(counts); free(displs);
  return 0;
}
