#include "utils_1d.h"
#include <limits.h>

/* Local comparisons of 1, 8, 16, 32, 64 and 128 steps favored 64 as a
 * compromise across domain sizes. Override at build time for another MPI
 * system, e.g. CPPFLAGS+=-DHEAT_HALO_STEPS=128 in student.mk. */
#ifndef HEAT_HALO_STEPS
#define HEAT_HALO_STEPS 64
#endif

#if HEAT_HALO_STEPS < 1 || HEAT_HALO_STEPS > INT_MAX
#error "HEAT_HALO_STEPS must be a positive MPI int count"
#endif

/* Spatial cache tile target, selected from 512, 1024, 2048, 4096 and 8192
 * points in local comparisons. Zero disables spatial blocking. */
#ifndef HEAT_TILE_POINTS
#define HEAT_TILE_POINTS 4096
#endif
#if HEAT_TILE_POINTS < 0 || HEAT_TILE_POINTS > INT_MAX
#error "HEAT_TILE_POINTS must be a nonnegative int"
#endif

typedef struct {
  double left;
  double right;
} BoundaryValues;

typedef struct {
  double *heat_array;
  double *err;
  size_t num_pts;
  double dx;
  BoundaryValues boundaries;
} HeatProblem;

typedef struct {
  int rank;
  int size;
} MpiContext;

typedef struct {
  double *current;
  double *next;
} TimeLevels;

/* A kernel receives a read/write view with disjoint buffers. Keep restrict
 * here, scoped to a single update, rather than on the time levels we swap. */
typedef struct {
  const double *restrict current;
  double *restrict next;
} UpdateBuffers;

/* Update [lo, hi], but accumulate error only for owned points [first, last]. */
typedef struct {
  size_t lo;
  size_t hi;
  size_t first;
  size_t last;
  int physical_left;
  int physical_right;
} UpdateRegion;

typedef struct {
  TimeLevels values;
  size_t points;
} TileWorkspace;

typedef struct {
  UpdateRegion region;
  int steps;
  double dx;
  BoundaryValues boundaries;
} TileUpdate;

/* Buffer layout: [padding][left halo][owned points][right halo][padding].
 * Only the owned interval [first, last] contributes to cumulative error. */
typedef struct {
  size_t count;
  size_t offset;
  size_t first;
  size_t last;
  int halo_steps;
  int left_rank;
  int right_rank;
  TimeLevels values;
  TileWorkspace tile;
  int *counts;
  int *displacements;
} LocalDomain;

static inline void swap_buffers(TimeLevels *values) {
  double *temporary = values->current;
  values->current = values->next;
  values->next = temporary;
}

static inline UpdateBuffers update_buffers(const TimeLevels *values) {
  return (UpdateBuffers){.current = values->current, .next = values->next};
}

/* The owned interval contributes to the error; the surrounding points only
 * supply dependencies for later steps. Separate allocations never overlap. */
static inline double update_range(UpdateBuffers buffers, UpdateRegion region, double dx) {
  const double *a = buffers.current;
  double *b = buffers.next;
  const size_t lo = region.lo, hi = region.hi;
  const size_t first = region.first, last = region.last;
  for (size_t j = lo; j < first; ++j)
    b[j] = 0.5 * (a[j-1] + a[j+1]);
  double sum = 0.0;
  for (size_t j = first; j <= last; ++j) {
    double value = 0.5 * (a[j-1] + a[j+1]);
    b[j] = value;
    /* Scale each contribution before summing: the unscaled total can
     * overflow even when the requested cumulative error is finite. */
    sum += dx * fabs(value - a[j]);
  }
  for (size_t j = last + 1; j <= hi; ++j)
    b[j] = 0.5 * (a[j-1] + a[j+1]);
  return sum;
}

/* Evaluate two consecutive explicit steps without storing a full intermediate
 * array. Every intermediate temperature still uses the original stencil, and
 * both time levels contribute to the error. Only the physical endpoints have
 * a fixed intermediate value; process boundaries use the received halos. */
static inline double update_pair(UpdateBuffers buffers, UpdateRegion region, double dx) {
  const double *a = buffers.current;
  double *b = buffers.next;
  const size_t lo = region.lo, hi = region.hi;
  const size_t first = region.first, last = region.last;
  const int physical_left = region.physical_left;
  const int physical_right = region.physical_right;
  double sum = 0.0;
  if (first == last && physical_left && physical_right) {
    double mid = 0.5 * (a[first-1] + a[first+1]);
    b[first] = mid;
    return dx * fabs(mid - a[first]);
  }
  for (size_t j = lo; j < first; ++j) {
    double left_mid = 0.5 * (a[j-2] + a[j]);
    double right_mid = 0.5 * (a[j] + a[j+2]);
    b[j] = 0.5 * (left_mid + right_mid);
  }
  size_t begin = first + (size_t)physical_left;
  size_t end = last - (size_t)physical_right;
  for (size_t j = begin; j <= end; ++j) {
    double left_mid = 0.5 * (a[j-2] + a[j]);
    double mid = 0.5 * (a[j-1] + a[j+1]);
    double right_mid = 0.5 * (a[j] + a[j+2]);
    double value = 0.5 * (left_mid + right_mid);
    b[j] = value;
    sum += dx * fabs(mid - a[j]) + dx * fabs(value - mid);
  }
  if (physical_left) {
    double mid = 0.5 * (a[first-1] + a[first+1]);
    double right_mid = 0.5 * (a[first] + a[first+2]);
    double value = 0.5 * (a[first-1] + right_mid);
    b[first] = value;
    sum += dx * fabs(mid - a[first]) + dx * fabs(value - mid);
  }
  if (physical_right) {
    double left_mid = 0.5 * (a[last-2] + a[last]);
    double mid = 0.5 * (a[last-1] + a[last+1]);
    double value = 0.5 * (left_mid + a[last+1]);
    b[last] = value;
    sum += dx * fabs(mid - a[last]) + dx * fabs(value - mid);
  }
  for (size_t j = last + 1; j <= hi; ++j) {
    double left_mid = 0.5 * (a[j-2] + a[j]);
    double right_mid = 0.5 * (a[j] + a[j+2]);
    b[j] = 0.5 * (left_mid + right_mid);
  }
  return sum;
}

/* Single-process path, also used when there are fewer points than ranks. */
static int solve_sequential(const HeatProblem *problem) {
  const size_t num_pts = problem->num_pts;
  double *scratch = malloc((num_pts + 2) * sizeof(double));
  if (!scratch) return -1;
  scratch[0] = problem->boundaries.left;
  scratch[num_pts + 1] = problem->boundaries.right;

  TimeLevels values = {.current = problem->heat_array, .next = scratch};
  const UpdateRegion region = {
    .lo = 1, .hi = num_pts, .first = 1, .last = num_pts,
    .physical_left = 1, .physical_right = 1
  };
  double sum = 0.0;
  int step = 0;
  for (; step + 1 < ITER; step += 2) {
    sum += update_pair(update_buffers(&values), region, problem->dx);
    swap_buffers(&values);
  }
  if (step < ITER) {
    sum += update_range(update_buffers(&values), region, problem->dx);
    swap_buffers(&values);
  }
  *problem->err = sum;
  if (values.current != problem->heat_array)
    memcpy(problem->heat_array + 1, values.current + 1, num_pts * sizeof(double));
  free(scratch);
  return 0;
}

static void destroy_domain(LocalDomain *domain) {
  free(domain->values.current);
  free(domain->values.next);
  free(domain->counts);
  free(domain->displacements);
  free(domain->tile.values.current);
  free(domain->tile.values.next);
}

/* Build the balanced partition and allocate the two local time levels.
 * The caller initializes domain to zero and releases it even on failure. */
static int create_domain(LocalDomain *domain, const HeatProblem *problem,
                         const MpiContext *mpi) {
  const size_t num_pts = problem->num_pts;
  const int rank = mpi->rank, size = mpi->size;
  const size_t base = num_pts / (size_t)size;
  const int rem = (int)(num_pts % (size_t)size);
  domain->count = base + (rank < rem);
  domain->offset = (size_t)rank * base + (size_t)(rank < rem ? rank : rem);
  domain->halo_steps = base < (size_t)HEAT_HALO_STEPS ? (int)base : HEAT_HALO_STEPS;
  domain->first = (size_t)domain->halo_steps + 1;
  domain->last = domain->first + domain->count - 1;
  domain->left_rank = rank ? rank - 1 : MPI_PROC_NULL;
  domain->right_rank = rank + 1 < size ? rank + 1 : MPI_PROC_NULL;

  const size_t buffer_count = domain->count + 2 * (size_t)domain->halo_steps + 2;
  if (buffer_count > SIZE_MAX / sizeof(double)) return -1;
  domain->values.current = malloc(buffer_count * sizeof(double));
  domain->values.next = malloc(buffer_count * sizeof(double));
  domain->counts = malloc((size_t)size * sizeof(int));
  domain->displacements = malloc((size_t)size * sizeof(int));
  if (!domain->values.current || !domain->values.next || !domain->counts || !domain->displacements)
    return -1;

#if HEAT_TILE_POINTS > 0
  if (domain->halo_steps > 1) {
    size_t minimum = 2 * (size_t)domain->halo_steps;
    size_t points = HEAT_TILE_POINTS > minimum ? HEAT_TILE_POINTS : minimum;
    if (domain->count > points) {
      size_t capacity = points + 2 * (size_t)domain->halo_steps + 2;
      if (capacity > SIZE_MAX / sizeof(double)) return -1;
      domain->tile.values.current = malloc(capacity * sizeof(double));
      domain->tile.values.next = malloc(capacity * sizeof(double));
      if (!domain->tile.values.current || !domain->tile.values.next) return -1;
      domain->tile.points = points;
    }
  }
#endif

  for (int r = 0; r < size; ++r) {
    domain->counts[r] = (int)(base + (r < rem));
    domain->displacements[r] = (int)((size_t)r * base + (size_t)(r < rem ? r : rem));
  }
  memcpy(domain->values.current + domain->first, problem->heat_array + domain->offset + 1,
         domain->count * sizeof(double));
  return 0;
}

static inline void set_boundaries(double *values, const LocalDomain *domain,
                                  BoundaryValues boundaries) {
  if (domain->left_rank == MPI_PROC_NULL)
    values[domain->first - 1] = boundaries.left;
  if (domain->right_rank == MPI_PROC_NULL)
    values[domain->last + 1] = boundaries.right;
}

static inline void start_halo_exchange(const LocalDomain *domain, double *current,
                                       MPI_Request requests[4]) {
  const int width = domain->halo_steps;
  const size_t halo = (size_t)width;
  MPI_Irecv(current + domain->first - halo, width, MPI_DOUBLE,
            domain->left_rank, 0, MPI_COMM_WORLD, &requests[0]);
  MPI_Irecv(current + domain->last + 1, width, MPI_DOUBLE,
            domain->right_rank, 1, MPI_COMM_WORLD, &requests[1]);
  MPI_Isend(current + domain->first, width, MPI_DOUBLE,
            domain->left_rank, 1, MPI_COMM_WORLD, &requests[2]);
  MPI_Isend(current + domain->last - halo + 1, width, MPI_DOUBLE,
            domain->right_rank, 0, MPI_COMM_WORLD, &requests[3]);
}

/* The input tile has steps valid halo points on each nonphysical side.
 * Shrink that interval as the explicit updates consume their dependencies. */
static inline double advance_tile(TimeLevels *values, const TileUpdate *update) {
  UpdateRegion region = update->region;
  double sum = 0.0;
  for (int step = 0; step < update->steps; ) {
    int steps = update->steps - step >= 2 ? 2 : 1;
    size_t halo = (size_t)(update->steps - step - steps);
    region.lo = region.physical_left ? region.first : region.first - halo;
    region.hi = region.physical_right ? region.last : region.last + halo;
    if (steps == 2)
      sum += update_pair(update_buffers(values), region, update->dx);
    else
      sum += update_range(update_buffers(values), region, update->dx);
    swap_buffers(values);
    step += steps;
  }
  return sum;
}

/* Balance tile lengths so even the last tile owns at least one halo width.
 * Thus a tile's halo cannot cross a physical boundary owned by another tile.
 * Every tile reads the same input time level and writes to the other buffer. */
static double advance_tiles(LocalDomain *domain, TimeLevels *values,
                            const TileUpdate *batch) {
  const size_t tiles = (domain->count - 1) / domain->tile.points + 1;
  const size_t base = domain->count / tiles;
  const size_t remainder = domain->count % tiles;
  const size_t halo = (size_t)batch->steps;
  double sum = 0.0;
  size_t start = domain->first;

  for (size_t tile = 0; tile < tiles; ++tile) {
    size_t count = base + (tile < remainder);
    size_t end = start + count - 1;
    TileUpdate update = {
      .region = {
        .first = halo + 1, .last = halo + count,
        .physical_left = batch->region.physical_left && tile == 0,
        .physical_right = batch->region.physical_right && tile + 1 == tiles
      },
      .steps = batch->steps, .dx = batch->dx, .boundaries = batch->boundaries
    };
    const UpdateRegion *region = &update.region;
    TimeLevels *scratch = &domain->tile.values;
    size_t copy_first = region->physical_left ? start - 1 : start - halo;
    size_t copy_last = region->physical_right ? end + 1 : end + halo;
    size_t destination = region->first - (start - copy_first);
    memcpy(scratch->current + destination, values->current + copy_first,
           (copy_last - copy_first + 1) * sizeof(double));
    if (region->physical_left) {
      scratch->current[region->first - 1] = batch->boundaries.left;
      scratch->next[region->first - 1] = batch->boundaries.left;
    }
    if (region->physical_right) {
      scratch->current[region->last + 1] = batch->boundaries.right;
      scratch->next[region->last + 1] = batch->boundaries.right;
    }
    sum += advance_tile(scratch, &update);
    memcpy(values->next + start, scratch->current + region->first,
           count * sizeof(double));
    start = end + 1;
  }
  swap_buffers(values);
  return sum;
}

/* Advance local time levels; wait for every request before swapping buffers. */
static double evolve_domain(LocalDomain *domain, const HeatProblem *problem) {
  TimeLevels values = domain->values;
  const double dx = problem->dx;
  const BoundaryValues boundaries = problem->boundaries;
  const size_t first = domain->first;
  const size_t last = domain->last;
  const int physical_left = domain->left_rank == MPI_PROC_NULL;
  const int physical_right = domain->right_rank == MPI_PROC_NULL;
  const UpdateRegion interior = {
    .lo = first + 1, .hi = last - 1,
    .first = first + 1, .last = last - 1
  };
  UpdateRegion left_edge = {.hi = first, .first = first, .last = first};
  UpdateRegion right_edge = {.lo = last, .first = last, .last = last};
  UpdateRegion owned = {
    .first = first, .last = last,
    .physical_left = physical_left, .physical_right = physical_right
  };
  double local_err = 0.0;

  for (int step = 0; step < ITER; ) {
    int chunk = ITER - step < domain->halo_steps ? ITER - step : domain->halo_steps;
    MPI_Request requests[4];
    start_halo_exchange(domain, values.current, requests);

    /* The first step's interior reads only owned points. It neither reads
     * the receive buffers nor modifies the send buffers while MPI is active.
     * Keep its output in next until the boundary work is complete. */
    if (domain->count > 2) {
      local_err += update_range(update_buffers(&values), interior, dx);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    set_boundaries(values.current, domain, boundaries);
    set_boundaries(values.next, domain, boundaries);

    size_t first_halo = (size_t)(chunk - 1);
    left_edge.lo = physical_left ? first : first - first_halo;
    right_edge.hi = physical_right ? last : last + first_halo;

    /* Complete the first step without recomputing the interior or counting
     * a single-point subdomain twice. Halo updates do not contribute error. */
    local_err += update_range(update_buffers(&values), left_edge, dx);
    if (domain->count > 1) {
      local_err += update_range(update_buffers(&values), right_edge, dx);
    }
    swap_buffers(&values);

    if (domain->tile.points && chunk > 1) {
      const TileUpdate batch = {
        .region = owned, .steps = chunk - 1, .dx = dx, .boundaries = boundaries
      };
      local_err += advance_tiles(domain, &values, &batch);
      step += chunk;
      continue;
    }

    for (int t = 1; t < chunk; ) {
      int steps = chunk - t >= 2 ? 2 : 1;
      size_t halo = (size_t)(chunk - t - steps);
      owned.lo = physical_left ? first : first - halo;
      owned.hi = physical_right ? last : last + halo;
      set_boundaries(values.next, domain, boundaries);
      if (steps == 2)
        local_err += update_pair(update_buffers(&values), owned, dx);
      else
        local_err += update_range(update_buffers(&values), owned, dx);
      swap_buffers(&values);
      t += steps;
    }
    step += chunk;
  }
  domain->values = values;
  return local_err;
}

static void gather_solution(const LocalDomain *domain, const HeatProblem *problem,
                            double local_err) {
  MPI_Gatherv(domain->values.current + domain->first, (int)domain->count, MPI_DOUBLE,
              problem->heat_array + 1, domain->counts, domain->displacements,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_err, problem->err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

/* MPI is owned by the driver. Return the final array and cumulative error on
 * rank 0, preserving the constant endpoints supplied in heat_array. */
int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts) {
  MpiContext mpi;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
  /* Gatherv uses int counts and displacements; array indexing uses size_t. */
  if (!heat_array || !err || !num_pts || num_pts > INT_MAX ||
      num_pts > SIZE_MAX / sizeof(double) - 2 ||
      (size_t)mpi.size > SIZE_MAX / sizeof(int)) return -1;

  const HeatProblem problem = {
    .heat_array = heat_array, .err = err, .num_pts = num_pts,
    .dx = 1.0 / (double)(num_pts + 1),
    .boundaries = {heat_array[0], heat_array[num_pts + 1]}
  };
  *err = 0.0;
  if (mpi.size == 1 || num_pts < (size_t)mpi.size)
    return mpi.rank == 0 ? solve_sequential(&problem) : 0;

  LocalDomain domain = {0};
  if (create_domain(&domain, &problem, &mpi) != 0) {
    destroy_domain(&domain);
    return -1;
  }
  double local_err = evolve_domain(&domain, &problem);
  gather_solution(&domain, &problem, local_err);
  destroy_domain(&domain);
  return 0;
}
