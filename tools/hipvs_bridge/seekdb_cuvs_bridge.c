/* seekdb bridge: exact brute-force kNN graph -> CAGRA index (C API only). */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hipblas/hipblas.h>
#include <rocprim/rocprim.hpp>
#include <limits.h>
#include <stdint.h>
#include <algorithm>
#ifndef __HIP_DEVICE_COMPILE__
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SK_X86_INTRINSICS 1
#endif
#endif
#include <atomic>
#include <thread>
#include <chrono>
#ifdef __linux__
#include <sys/mman.h>
#endif

/* ---- env-gated profiling (TILED_TIME=1): host-time accumulators, ms ---- */
static int g_time = 0;
static double g_t_copyin = 0, g_t_gemm = 0, g_t_fin = 0, g_t_tsel = 0,
              g_t_fsel = 0, g_t_gcopy = 0, g_t_import = 0, g_t_total = 0;
static long g_n_gemm = 0;

static inline double sk_now_ms()
{
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void sk_time_report(const char* tag)
{
  if (!g_time) return;
  fprintf(stderr,
          "[time] %s total=%.1f gemm=%.1f (n=%ld, %.2f ms/call) fin=%.1f tsel=%.1f "
          "fsel=%.1f gcopy=%.1f import=%.1f copyin=%.1f other=%.1f ms\n",
          tag, g_t_total, g_t_gemm, g_n_gemm,
          g_n_gemm ? g_t_gemm / (double)g_n_gemm : 0.0,
          g_t_fin, g_t_tsel, g_t_fsel, g_t_gcopy, g_t_import, g_t_copyin,
          g_t_total - (g_t_gemm + g_t_fin + g_t_tsel + g_t_fsel + g_t_gcopy +
                       g_t_import + g_t_copyin));
}

typedef struct {
  cuvsResources_t res;
  cuvsCagraIndex_t idx;
  long n, dim, k;
  void* d_dataset;
  void* d_graph;
  float* host_base;
  uint32_t* host_graph;
  long graph_k;
  long index_k;
  int fp16;
  uint32_t* stamp;
  uint32_t stamp_gen;
  void* pinned_base;
} sk_bf_index;

/* Large host buffers (dataset copy, neighbour graph) are tens to hundreds of
   megabytes and are written once, front to back.  Faulting them in 4 KiB at a
   time dominated the host side of the build, so map them huge-page backed and
   populate them up front. */
static void* sk_big_alloc(size_t bytes)
{
#ifdef __linux__
  void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return NULL;
  madvise(p, bytes, MADV_HUGEPAGE);
  return p;
#else
  return malloc(bytes);
#endif
}

static void sk_big_free(void* p, size_t bytes)
{
  if (!p) return;
#ifdef __linux__
  munmap(p, bytes);
#else
  (void)bytes;
  free(p);
#endif
}

static int env_int(const char* key, int dflt)
{
  const char* v = getenv(key);
  return v && *v ? atoi(v) : dflt;
}

static std::atomic<int> g_mem_phase{0};
static std::atomic<long long> g_mem_peak{0};
static std::atomic<int> g_mem_sampling{0};
static std::atomic<int> g_mem_peak_phase{0};

static inline uint16_t f2h(float f)
{
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  uint32_t expo = (x >> 23) & 0xFFu;
  uint32_t mant = x & 0x7FFFFFu;
  if (expo == 0xFFu) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
  int e = (int)expo - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    int shift = 14 - e;
    uint32_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
  return (uint16_t)(sign | h);
}

static long long mem_used_bytes()
{
  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return -1;
  return (long long)(total_b - free_b);
}

static void memdbg(int phase, const char* tag)
{
  if (!env_int("TILED_MEMDBG", 0)) return;
  g_mem_phase.store(phase, std::memory_order_relaxed);
  fprintf(stderr, "[memdbg] phase=%d %-24s used=%.3f GB\n", phase, tag,
          (double)mem_used_bytes() / 1e9);
}

static void mem_sampler_start()
{
  if (!env_int("TILED_MEMDBG", 0)) return;
  g_mem_sampling.store(1);
  std::thread([]() {
    long long best = 0; int best_phase = 0;
    while (g_mem_sampling.load(std::memory_order_relaxed)) {
      long long u = mem_used_bytes();
      if (u > best) { best = u; best_phase = g_mem_phase.load(std::memory_order_relaxed); }
      std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
    g_mem_peak.store(best, std::memory_order_relaxed);
    g_mem_peak_phase.store(best_phase, std::memory_order_relaxed);
    fprintf(stderr, "[memdbg] sampled peak=%.3f GB at phase=%d\n", (double)best / 1e9, best_phase);
  }).detach();
}

static void mem_sampler_stop()
{
  if (!env_int("TILED_MEMDBG", 0)) return;
  g_mem_sampling.store(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
}

struct Ranked { float d; unsigned id; };
static inline bool ranked_less(const Ranked& a, const Ranked& b)
{
  return (a.d < b.d) || (a.d == b.d && a.id < b.id);
}
static inline float l2_dist_scalar(const float* q, const float* x, long dim)
{
  float d = 0.0f;
  for (long t = 0; t < dim; ++t) { float z = q[t] - x[t]; d = fmaf(z, z, d); }
  return d;
}
#ifdef SK_X86_INTRINSICS
__attribute__((target("avx2,fma")))
static float l2_dist_avx2(const float* q, const float* x, long dim)
{
  __m256 acc = _mm256_setzero_ps();
  long t = 0;
  for (; t + 8 <= dim; t += 8) {
    __m256 a = _mm256_loadu_ps(q + t);
    __m256 b = _mm256_loadu_ps(x + t);
    __m256 z = _mm256_sub_ps(a, b);
    acc = _mm256_fmadd_ps(z, z, acc);
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sm = _mm_add_ps(lo, hi);
  sm = _mm_hadd_ps(sm, sm);
  sm = _mm_hadd_ps(sm, sm);
  float d = _mm_cvtss_f32(sm);
  for (; t < dim; ++t) { float z = q[t] - x[t]; d = fmaf(z, z, d); }
  return d;
}
static int g_have_avx2 = -1;
static inline float l2_dist(const float* q, const float* x, long dim)
{
  if (g_have_avx2 < 0) g_have_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
  return g_have_avx2 ? l2_dist_avx2(q, x, dim) : l2_dist_scalar(q, x, dim);
}
#else
static inline float l2_dist(const float* q, const float* x, long dim) { return l2_dist_scalar(q, x, dim); }
#endif

static void set_tensor(DLManagedTensor* t, void* data, DLDeviceType dev, int bits,
                       long r, long c, long* shape)
{
  memset(t, 0, sizeof(*t));
  shape[0] = r; shape[1] = c;
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = dev;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = kDLFloat;
  t->dl_tensor.dtype.bits = bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = 0;
  t->dl_tensor.byte_offset = 0;
}

__global__ static void fill_pad_kernel(float* x, long start, long count, float v)
{
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) x[start + i] = v;
}

__global__ static void row_norm_kernel(const float* x, float* norms, long n, long dim)
{
  long row = (long)blockIdx.x;
  int t = threadIdx.x;
  __shared__ float smem[256];
  float v = 0.0f;
  for (long j = t; j < dim; j += blockDim.x) {
    float a = x[row * dim + j];
    v = fmaf(a, a, v);
  }
  smem[t] = v;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (t < stride) smem[t] += smem[t + stride];
    __syncthreads();
  }
  if (t == 0) norms[row] = smem[0];
}

__global__ static void finish_distance_tile(float* d,
                                             const float* norms,
                                             long n, long q0, long x0,
                                             int B, int T)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)B * T;
  if (i >= count) return;
  long qr = (long)(i / T);
  long xc = (long)(i % T);
  long qid = q0 + qr;
  long xid = x0 + xc;
  if (qid >= n || xid >= n) { d[i] = INFINITY; return; }
  d[i] += norms[qid] + norms[xid];
}

__global__ static void gather_tile_topk(const float* sorted_keys,
                                         const uint16_t* sorted_ids,
                                         float* candidates,
                                         uint32_t* candidate_ids,
                                         int B, int T, int K, int tile_no,
                                         int ntiles, int tile_count)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B) return;
  int m = tile_count < K ? tile_count : K;
  size_t src = (size_t)q * T;
  size_t dst = ((size_t)q * ntiles + tile_no) * K;
  for (int j = 0; j < m; ++j) {
    candidates[dst + j] = sorted_keys[src + j];
    uint16_t local = sorted_ids[src + j];
    candidate_ids[dst + j] = local == UINT16_MAX ? UINT_MAX : (uint32_t)(tile_no * T + local);
  }
  for (int j = m; j < K; ++j) {
    candidates[dst + j] = INFINITY;
    candidate_ids[dst + j] = UINT_MAX;
  }
}

__global__ static void gather_final_topk(const uint32_t* sorted_ids,
                                          uint32_t* graph_block,
                                          int B, int K, int out_k,
                                          int candidates_per_row)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= B) return;
  size_t src = (size_t)q * candidates_per_row;
  size_t dst = (size_t)q * out_k;
  for (int j = 0; j < out_k; ++j) {
    uint32_t id = sorted_ids[src + j];
    graph_block[dst + j] = id == UINT_MAX ? 0u : id;
  }
}

static int gcd_int(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

// ---------------- custom FP32 distance-tile GEMM (replaces hipBLAS SGEMM) ----------------
#define DT_BM 64
#define DT_BN 64
#define DT_BK 16
__global__ static void dist_tile_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ X,
                                        const float* __restrict__ norms,
                                        float* __restrict__ D,
                                        int dim, long q0, long x0,
                                        int B, int T, long n)
{
  __shared__ float Qs[DT_BM][DT_BK + 1];
  __shared__ float Xs[DT_BN][DT_BK + 1];
  const int tx = threadIdx.x & 15;
  const int ty = threadIdx.x >> 4;
  const int b0 = blockIdx.x * DT_BM;
  const int t0 = blockIdx.y * DT_BN;
  const int ldr = threadIdx.x >> 2;
  const int ldc = (threadIdx.x & 3) * 4;
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < dim; k0 += DT_BK) {
    const float* qsrc = Q + (size_t)(q0 + b0 + ldr) * dim + k0 + ldc;
    float4 qv = *(const float4*)qsrc;
    Qs[ldr][ldc + 0] = qv.x; Qs[ldr][ldc + 1] = qv.y;
    Qs[ldr][ldc + 2] = qv.z; Qs[ldr][ldc + 3] = qv.w;
    const float* xsrc = X + (size_t)(x0 + t0 + ldr) * dim + k0 + ldc;
    float4 xv = *(const float4*)xsrc;
    Xs[ldr][ldc + 0] = xv.x; Xs[ldr][ldc + 1] = xv.y;
    Xs[ldr][ldc + 2] = xv.z; Xs[ldr][ldc + 3] = xv.w;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < DT_BK; ++k) {
      float a0 = Qs[tx * 4 + 0][k], a1 = Qs[tx * 4 + 1][k];
      float a2 = Qs[tx * 4 + 2][k], a3 = Qs[tx * 4 + 3][k];
      float c0 = Xs[ty * 4 + 0][k], c1 = Xs[ty * 4 + 1][k];
      float c2 = Xs[ty * 4 + 2][k], c3 = Xs[ty * 4 + 3][k];
      acc[0][0] = fmaf(a0, c0, acc[0][0]); acc[0][1] = fmaf(a0, c1, acc[0][1]);
      acc[0][2] = fmaf(a0, c2, acc[0][2]); acc[0][3] = fmaf(a0, c3, acc[0][3]);
      acc[1][0] = fmaf(a1, c0, acc[1][0]); acc[1][1] = fmaf(a1, c1, acc[1][1]);
      acc[1][2] = fmaf(a1, c2, acc[1][2]); acc[1][3] = fmaf(a1, c3, acc[1][3]);
      acc[2][0] = fmaf(a2, c0, acc[2][0]); acc[2][1] = fmaf(a2, c1, acc[2][1]);
      acc[2][2] = fmaf(a2, c2, acc[2][2]); acc[2][3] = fmaf(a2, c3, acc[2][3]);
      acc[3][0] = fmaf(a3, c0, acc[3][0]); acc[3][1] = fmaf(a3, c1, acc[3][1]);
      acc[3][2] = fmaf(a3, c2, acc[3][2]); acc[3][3] = fmaf(a3, c3, acc[3][3]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int qi = b0 + tx * 4 + i;
    const long qid = q0 + qi;
    const float nq = norms[qid];
    float* drow = D + (size_t)qi * T + t0 + ty * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const long xid = x0 + t0 + ty * 4 + j;
      float v = fmaf(-2.0f, acc[i][j], nq + norms[xid]);
      drow[j] = (qid >= n || xid >= n) ? INFINITY : v;
    }
  }
}

// ---------------- custom top-K selection (replaces rocPRIM sorts) ----------------
__global__ static void select_topk_kernel(const float* __restrict__ keys,
                                          const uint32_t* __restrict__ ids,
                                          int stride, int C, int K,
                                          long base_id, int out_stride,
                                          float* __restrict__ out_keys,
                                          uint32_t* __restrict__ out_ids)
{
  const int row = blockIdx.x;
  const float* krow = keys + (size_t)row * stride;
  const uint32_t* irow = ids ? (ids + (size_t)row * stride) : nullptr;
  __shared__ uint32_t hist[8][256];
  __shared__ float sk[256];
  __shared__ uint32_t si[256];
  __shared__ uint32_t s_prefix, s_rank, s_less, cnt_less, cnt_eq;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;

  if (C <= K) {
    for (int i = tid; i < K; i += blockDim.x) {
      float v = i < C ? krow[i] : INFINITY;
      uint32_t id = i < C ? (irow ? irow[i] : (uint32_t)(base_id + i)) : 0u;
      out_keys[(size_t)row * out_stride + i] = v;
      out_ids[(size_t)row * out_stride + i] = id;
    }
    return;
  }

  if (tid == 0) { s_prefix = 0u; s_rank = (uint32_t)K; }
  __syncthreads();
  uint32_t pmask = 0u;
  for (int pass = 3; pass >= 0; --pass) {
    for (int i = tid; i < 8 * 256; i += blockDim.x) ((uint32_t*)hist)[i] = 0u;
    __syncthreads();
    uint32_t pm = pmask, pf = s_prefix;
    int shift = pass * 8;
    for (int i = tid; i < C; i += blockDim.x) {
      uint32_t u = __float_as_uint(krow[i]);
      if ((u & pm) == pf) atomicAdd(&hist[warp][(u >> shift) & 0xFF], 1u);
    }
    __syncthreads();
    if (tid < 256) {
      uint32_t c = 0u;
      #pragma unroll
      for (int w = 0; w < 8; ++w) c += hist[w][tid];
      hist[0][tid] = c;
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t cum = 0u, r = s_rank, chosen = 0u;
      for (int b = 0; b < 256; ++b) {
        uint32_t c = hist[0][b];
        if (cum + c >= r) { chosen = (uint32_t)b; s_rank = r - cum; s_less = cum; break; }
        cum += c;
      }
      s_prefix |= chosen << shift;
    }
    pmask |= 0xFFu << shift;
    __syncthreads();
  }

  const uint32_t pref = s_prefix;
  const uint32_t need_eq = s_rank;
  const uint32_t need_less = (uint32_t)K - need_eq;
  if (tid == 0) { cnt_less = 0u; cnt_eq = 0u; }
  __syncthreads();
  for (int i = tid; i < C; i += blockDim.x) {
    uint32_t u = __float_as_uint(krow[i]);
    if (u < pref) {
      uint32_t p = atomicAdd(&cnt_less, 1u);
      if (p < need_less) {
        sk[p] = krow[i];
        si[p] = irow ? irow[i] : (uint32_t)(base_id + i);
      }
    } else if (u == pref) {
      uint32_t p = atomicAdd(&cnt_eq, 1u);
      if (p < need_eq) {
        sk[need_less + p] = krow[i];
        si[need_less + p] = irow ? irow[i] : (uint32_t)(base_id + i);
      }
    }
  }
  __syncthreads();

  for (int k = 2; k <= K; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < K; i += blockDim.x) {
        int ixj = i ^ j;
        if (ixj > i) {
          bool asc = ((i & k) == 0);
          float a = sk[i], b = sk[ixj];
          uint32_t ia = si[i], ib = si[ixj];
          bool gt = (b < a) || (b == a && ib < ia);
          bool lt = (a < b) || (a == b && ia < ib);
          if (asc ? gt : lt) { sk[i] = b; si[i] = ib; sk[ixj] = a; si[ixj] = ia; }
        }
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < K; i += blockDim.x) {
    out_keys[(size_t)row * out_stride + i] = sk[i];
    out_ids[(size_t)row * out_stride + i] = si[i];
  }
}

/* ---------------- fused finish + exact top-K tile select ----------------
 * The per-column norm correction of finish_distance_tile is folded into the
 * key computation (key = dot + (nq + nx), bit-identical arithmetic), removing
 * one full read+write of the tile.
 *
 * Fast path: a 1/28 strided sample of the row yields the (STK_EST_E *
 * K / Cv)-th order statistic of the distance distribution via two narrowing
 * 11+11 bit shared histograms; one collect pass then gathers every element at
 * or below that threshold (~STK_EST_E of them, a handful of shared atomics per
 * thread instead of one per element, which is what the old first pass cost),
 * and an in-shared 30-bit radix select picks the exact K smallest of that set.
 *
 * Fallback: if the estimate is off (too few candidates, or a degenerate row
 * overflows the candidate buffer) the previous exact scheme runs instead: two
 * narrowing passes over the row (11 + 11 high bits) resolve a 22-bit prefix,
 * a collect pass sends the matching candidates to shared memory where the last
 * 10 bits are resolved, and degenerate rows take two more global passes.
 * Both paths emit the same K smallest, so the tile output is unchanged. */
#define STK_NT 256
#define STK_HB 2048
#define STK_CAP 1024
#define STK_SAMP 1024
#define STK_MB 1024
#define STK_EST_E 640
#define STK_P1_BITS 11
#define STK_LO_BITS 10
#define STK_P1_SHIFT (32 - STK_P1_BITS)

__device__ static void stk_pick(uint32_t* hist, uint32_t* part, int nb, int shift,
                                uint32_t rank, uint32_t pat, uint32_t less_base,
                                uint32_t* out_pat, uint32_t* out_less, uint32_t* out_rank,
                                int tid)
{
  const int per = nb / STK_NT;
  uint32_t mysum = 0;
  for (int j = 0; j < per; ++j) mysum += hist[tid * per + j];
  part[tid] = mysum;
  __syncthreads();
  for (int off = 1; off < STK_NT; off <<= 1) {
    uint32_t v = (tid >= off) ? part[tid - off] : 0u;
    __syncthreads();
    part[tid] += v;
    __syncthreads();
  }
  uint32_t excl = part[tid] - mysum;
  if (rank > excl && rank <= excl + mysum) {
    uint32_t cum = excl;
    for (int j = 0; j < per; ++j) {
      uint32_t c = hist[tid * per + j];
      if (cum + c >= rank) {
        *out_pat = pat | ((uint32_t)(tid * per + j) << shift);
        *out_less = less_base + cum;
        *out_rank = rank - cum;
        break;
      }
      cum += c;
    }
  }
  __syncthreads();
}

/* body sees i (int), v (float key), u (uint32 bits of v) */
#define STK_FOR_EACH(BODY)                                                              \
  {                                                                                     \
    const int _vend = Cv & ~3;                                                          \
    int _i = tid * 4;                                                                   \
    for (; _i < _vend; _i += STK_NT * 4) {                                              \
      float4 _d = *(const float4*)(drow + _i);                                          \
      float4 _n = RAW ? *(const float4*)(nx + _i) : make_float4(0.f, 0.f, 0.f, 0.f);     \
      float _k0 = RAW ? (_d.x + (nq + _n.x)) : _d.x;                                     \
      float _k1 = RAW ? (_d.y + (nq + _n.y)) : _d.y;                                     \
      float _k2 = RAW ? (_d.z + (nq + _n.z)) : _d.z;                                     \
      float _k3 = RAW ? (_d.w + (nq + _n.w)) : _d.w;                                     \
      { const int i = _i; const float v = _k0; const uint32_t u = __float_as_uint(v); BODY } \
      { const int i = _i + 1; const float v = _k1; const uint32_t u = __float_as_uint(v); BODY } \
      { const int i = _i + 2; const float v = _k2; const uint32_t u = __float_as_uint(v); BODY } \
      { const int i = _i + 3; const float v = _k3; const uint32_t u = __float_as_uint(v); BODY } \
    }                                                                                   \
    for (int _j = _vend + tid; _j < Cv; _j += STK_NT) {                                  \
      const int i = _j;                                                                  \
      const float v = RAW ? (drow[_j] + (nq + nx[_j])) : drow[_j];                        \
      const uint32_t u = __float_as_uint(v);                                             \
      BODY                                                                              \
    }                                                                                   \
  }

template <bool RAW>
__global__ static void select_tile_fused_kernel(const float* __restrict__ dots,
                                                const float* __restrict__ norms,
                                                long x0, int Cv, int stride,
                                                int K, long q0,
                                                float* __restrict__ out_keys,
                                                uint32_t* __restrict__ out_ids,
                                                int out_stride)
{
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* drow = dots + (size_t)row * stride;
  const float* nx = norms + x0;
  const float nq = RAW ? norms[q0 + row] : 0.0f;

  __shared__ uint32_t hist[STK_HB];
  __shared__ uint32_t part[STK_NT];
  __shared__ float ck[STK_CAP];
  __shared__ uint32_t ci[STK_CAP];
  __shared__ float sk[STK_NT];
  __shared__ uint32_t si[STK_NT];
  __shared__ uint32_t s_pat, s_less, s_rank, cnt_c, cnt_ovf, cnt_l;

  float* ok = out_keys + (size_t)row * out_stride;
  uint32_t* oi = out_ids + (size_t)row * out_stride;

  if (Cv <= K || Cv <= 0) {
    for (int i = tid; i < K; i += STK_NT) {
      float v = INFINITY;
      uint32_t id = 0u;
      if (i < Cv) {
        v = RAW ? (drow[i] + (nq + nx[i])) : drow[i];
        id = (uint32_t)(x0 + i);
      }
      ok[i] = v;
      oi[i] = id;
    }
    return;
  }

  /* ---- sample-based threshold estimate -------------------------------------
   * A 1/28 strided sample of the row (32-float chunks, so every fetched line is
   * fully used) gives the (K*STK_EST_E/Cv) order statistic of the distance
   * distribution; the count of row elements at or below that threshold is then
   * ~STK_EST_E.  Only the sample pays shared-memory atomics, and one collect
   * pass over the row gathers every element <= threshold.  If the collected
   * set holds at least K elements it contains the exact K smallest (any value
   * below the K-th collected element is necessarily collected too), so the
   * in-shared rank select is exact.  Pathological rows fall back to the
   * histogram passes below. */
  {
    const int ncch = STK_SAMP / 32;
    const int cstep = (Cv + ncch - 1) / ncch;
    const int lane = tid & 31;
    long lr = (long)STK_EST_E * STK_SAMP / (long)Cv;
    uint32_t q = (uint32_t)(lr < 1 ? 1 : (lr > STK_SAMP ? STK_SAMP : lr));
    for (int i = tid; i < STK_MB; i += STK_NT) hist[i] = 0u;
    if (tid == 0) { s_pat = 0u; s_less = 0u; s_rank = q; }
    __syncthreads();
    for (int c = tid >> 5; c < ncch; c += STK_NT / 32) {
      int idx = c * cstep + lane;
      if (idx < Cv) {
        float v = RAW ? (drow[idx] + (nq + nx[idx])) : drow[idx];
        atomicAdd(&hist[(__float_as_uint(v) >> 22) & (STK_MB - 1u)], 1u);
      }
    }
    __syncthreads();
    {
      uint32_t pat = 0u, less = 0u, rank = q;
      stk_pick(hist, part, STK_MB, 22, rank, pat, less, &s_pat, &s_less, &s_rank, tid);
    }
    {
      const uint32_t pat1 = s_pat;
      for (int i = tid; i < STK_MB; i += STK_NT) hist[i] = 0u;
      __syncthreads();
      for (int c = tid >> 5; c < ncch; c += STK_NT / 32) {
        int idx = c * cstep + lane;
        if (idx < Cv) {
          float v = RAW ? (drow[idx] + (nq + nx[idx])) : drow[idx];
          uint32_t u = __float_as_uint(v);
          if ((u >> 22) == (pat1 >> 22)) atomicAdd(&hist[(u >> 12) & (STK_MB - 1u)], 1u);
        }
      }
      __syncthreads();
      {
        uint32_t pat = pat1, less = s_less, rank = s_rank;
        stk_pick(hist, part, STK_MB, 12, rank, pat, less, &s_pat, &s_less, &s_rank, tid);
      }
    }
  }
  const uint32_t ub = s_pat | 0xFFFu;   /* upper edge of the estimate bucket */
  if (tid == 0) { cnt_c = 0u; cnt_ovf = 0u; }
  __syncthreads();
  STK_FOR_EACH({
    if (u <= ub) {
      uint32_t p = atomicAdd(&cnt_c, 1u);
      if (p < (uint32_t)STK_CAP) { ck[p] = v; ci[p] = (uint32_t)(x0 + i); }
      else atomicAdd(&cnt_ovf, 1u);
    }
  });
  __syncthreads();
  const uint32_t ncand = cnt_c;
  if (ncand >= (uint32_t)K && cnt_ovf == 0u) {
    /* ---- exact K smallest of the collected set (in shared memory) ---- */
    uint32_t mpat = 0u, mless = 0u, mrank = (uint32_t)K, him = 0u;
    #pragma unroll
    for (int round = 0; round < 3; ++round) {
      const int rshift = 22 - round * 10;
      for (int i = tid; i < STK_MB; i += STK_NT) hist[i] = 0u;
      __syncthreads();
      for (uint32_t i = tid; i < ncand; i += STK_NT) {
        uint32_t u = __float_as_uint(ck[i]);
        if ((u & him) == (mpat & him)) atomicAdd(&hist[(u >> rshift) & (STK_MB - 1u)], 1u);
      }
      __syncthreads();
      {
        uint32_t pat = mpat, less = mless, rank = mrank;
        stk_pick(hist, part, STK_MB, rshift, rank, pat, less, &s_pat, &s_less, &s_rank, tid);
      }
      mpat = s_pat; mless = s_less; mrank = s_rank;
      him |= 0x3FFu << rshift;
    }
    const uint32_t bpat = mpat;
    const uint32_t less_total = mless;
    const uint32_t need_eq = mrank;
    if (tid == 0) { cnt_c = 0u; cnt_ovf = 0u; }
    __syncthreads();
    for (uint32_t i = tid; i < ncand; i += STK_NT) {
      uint32_t u = __float_as_uint(ck[i]);
      if ((u & him) == (bpat & him)) {
        uint32_t p = atomicAdd(&cnt_c, 1u);
        if (p < need_eq) { sk[less_total + p] = ck[i]; si[less_total + p] = ci[i]; }
      } else if (u < bpat) {
        uint32_t p = atomicAdd(&cnt_ovf, 1u);
        if (p < less_total) { sk[p] = ck[i]; si[p] = ci[i]; }
      }
    }
    __syncthreads();
    for (int i = tid; i < K; i += STK_NT) { ok[i] = sk[i]; oi[i] = si[i]; }
    return;
  }

  if (tid == 0) { s_pat = 0u; s_less = 0u; s_rank = (uint32_t)K; }
  for (int i = tid; i < STK_HB; i += STK_NT) hist[i] = 0u;
  __syncthreads();
  STK_FOR_EACH({ atomicAdd(&hist[u >> STK_P1_SHIFT], 1u); });
  __syncthreads();
  {
    uint32_t pat = s_pat, less = s_less, rank = s_rank;
    stk_pick(hist, part, STK_HB, STK_P1_SHIFT, rank, pat, less,
             &s_pat, &s_less, &s_rank, tid);
  }

  const uint32_t m1 = ~((1u << STK_P1_SHIFT) - 1u);
  for (int i = tid; i < STK_HB; i += STK_NT) hist[i] = 0u;
  __syncthreads();
  {
    const uint32_t pat = s_pat;
    STK_FOR_EACH({
      if ((u & m1) == pat) atomicAdd(&hist[(u >> STK_LO_BITS) & (STK_HB - 1u)], 1u);
    });
  }
  __syncthreads();
  {
    uint32_t pat = s_pat, less = s_less, rank = s_rank;
    stk_pick(hist, part, STK_HB, STK_LO_BITS, rank, pat, less,
             &s_pat, &s_less, &s_rank, tid);
  }

  /* collect pass */
  const uint32_t m2 = ~((1u << STK_LO_BITS) - 1u);
  const uint32_t lopmask = (1u << STK_LO_BITS) - 1u;
  const uint32_t pat2 = s_pat;
  const uint32_t need_eq = s_rank;
  const uint32_t need_less = (uint32_t)K - need_eq;
  if (tid == 0) { cnt_c = 0u; cnt_ovf = 0u; cnt_l = 0u; }
  __syncthreads();
  STK_FOR_EACH({
    if ((u & m2) == pat2) {
      uint32_t p = atomicAdd(&cnt_c, 1u);
      if (p < (uint32_t)STK_CAP) { ck[p] = v; ci[p] = (uint32_t)(x0 + i); }
      else atomicAdd(&cnt_ovf, 1u);
    } else if (u < pat2) {
      uint32_t p = atomicAdd(&cnt_l, 1u);
      if (p < need_less) { sk[p] = v; si[p] = (uint32_t)(x0 + i); }
    }
  });
  __syncthreads();
  const uint32_t n_c = cnt_c;
  const uint32_t ovf = cnt_ovf;

  if (ovf == 0u) {
    for (int i = tid; i < (1 << STK_LO_BITS); i += STK_NT) hist[i] = 0u;
    __syncthreads();
    for (uint32_t i = tid; i < n_c; i += STK_NT)
      atomicAdd(&hist[__float_as_uint(ck[i]) & lopmask], 1u);
    __syncthreads();
    if (tid == 0) { s_pat = 0u; s_less = 0u; s_rank = need_eq; }
    __syncthreads();
    {
      uint32_t pat = s_pat, less = s_less, rank = s_rank;
      stk_pick(hist, part, 1 << STK_LO_BITS, 0, rank, pat, less,
               &s_pat, &s_less, &s_rank, tid);
    }
    const uint32_t lop = s_pat;
    const uint32_t lo_less = s_less;
    const uint32_t lo_eq = s_rank;
    if (tid == 0) { cnt_l = 0u; cnt_c = 0u; }
    __syncthreads();
    for (uint32_t i = tid; i < n_c; i += STK_NT) {
      uint32_t lu = __float_as_uint(ck[i]) & lopmask;
      if (lu < lop) {
        uint32_t p = atomicAdd(&cnt_l, 1u);
        if (p < lo_less) { sk[need_less + p] = ck[i]; si[need_less + p] = ci[i]; }
      } else if (lu == lop) {
        uint32_t p = atomicAdd(&cnt_c, 1u);
        if (p < lo_eq) { sk[need_less + lo_less + p] = ck[i]; si[need_less + lo_less + p] = ci[i]; }
      }
    }
  } else {
    for (int i = tid; i < (1 << STK_LO_BITS); i += STK_NT) hist[i] = 0u;
    __syncthreads();
    {
      const uint32_t p2 = pat2;
      STK_FOR_EACH({
        if ((u & m2) == p2) atomicAdd(&hist[u & lopmask], 1u);
      });
    }
    __syncthreads();
    if (tid == 0) { s_less = 0u; s_rank = need_eq; }
    __syncthreads();
    {
      uint32_t pat = pat2, less = 0u, rank = need_eq;
      stk_pick(hist, part, 1 << STK_LO_BITS, 0, rank, pat, less,
               &s_pat, &s_less, &s_rank, tid);
    }
    const uint32_t pat3 = s_pat | (pat2 & m2);
    const uint32_t less3 = s_less;
    const uint32_t eq3 = s_rank;
    if (tid == 0) { cnt_l = 0u; cnt_c = 0u; }
    __syncthreads();
    const uint32_t base3 = need_less + less3;
    STK_FOR_EACH({
      if (u < pat3) {
        uint32_t p = atomicAdd(&cnt_l, 1u);
        if (p < base3) { sk[p] = v; si[p] = (uint32_t)(x0 + i); }
      } else if (u == pat3) {
        uint32_t p = atomicAdd(&cnt_c, 1u);
        if (p < eq3) { sk[base3 + p] = v; si[base3 + p] = (uint32_t)(x0 + i); }
      }
    });
  }
  __syncthreads();
  for (int i = tid; i < K; i += STK_NT) {
    ok[i] = sk[i];
    oi[i] = si[i];
  }
}

static int sk_finish_graph_copy(uint32_t* graph, long K, uint32_t* stage,
                                hipEvent_t ev, long q0, int rows)
{
  if (hipEventSynchronize(ev) != hipSuccess) return 1;
  memcpy(graph + (size_t)q0 * K, stage, (size_t)rows * K * sizeof(uint32_t));
  return 0;
}

static int build_exact_tiled_graph(const float* base, long n, long dim, int K,
                                   int B, int T, uint32_t** graph_out)
{
  int rc = 1;
  float* d_base = NULL;
  float* d_norms = NULL;
  float* d_dist = NULL;
  float* cand_k = NULL;
  uint32_t* cand_id = NULL;
  float* d_graph_k = NULL;
  uint32_t* d_graph_block[2];
  uint32_t* stage[2];
  int pend_buf[2];
  long pend_q0[2];
  int pend_rows[2];
  int copy_npend = 0;
  int copy_gen = 0;
  hipStream_t s_work = NULL;
  hipStream_t s_copy = NULL;
  hipEvent_t ev_copy[2];
  hipEvent_t ev_final = NULL;
  int ev_ready = 0;
  hipblasHandle_t blas = NULL;
  int use_blas = env_int("TILED_GEMM_BLAS", dim == 256 ? 1 : 0);
  uint32_t* graph = NULL;
  long lcm = (long)B * (long)T / gcd_int(B, T);
  long padded_n = ((n + lcm - 1) / lcm) * lcm;
  int ntiles = (int)(padded_n / T);
  int cpr = ntiles * K;
  size_t tile_elems = (size_t)B * T;
  size_t candidate_elems = (size_t)B * cpr;
  double _t0 = 0, _t_fn = 0;

  d_graph_block[0] = d_graph_block[1] = NULL;
  stage[0] = stage[1] = NULL;
  pend_buf[0] = pend_buf[1] = 0;
  pend_q0[0] = pend_q0[1] = 0;
  pend_rows[0] = pend_rows[1] = 0;
  ev_copy[0] = ev_copy[1] = NULL;

  memdbg(10, "bf_enter");
  g_time = env_int("TILED_TIME", 0);
  g_t_copyin = g_t_gemm = g_t_fin = g_t_tsel = g_t_fsel = g_t_gcopy = 0;
  g_n_gemm = 0;
  _t_fn = sk_now_ms();
  if (n <= 0 || dim <= 0 || K <= 0 || K > 256 || (K & (K - 1)) != 0 || B <= 0 || T <= 0 ||
      padded_n > INT_MAX || tile_elems > UINT_MAX || candidate_elems > UINT_MAX) return 1;
  if (hipMalloc((void**)&d_base, (size_t)padded_n * dim * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_norms, (size_t)padded_n * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_dist, tile_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&cand_k, candidate_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&cand_id, candidate_elems * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&d_graph_k, (size_t)B * K * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_graph_block[0], (size_t)B * K * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&d_graph_block[1], (size_t)B * K * sizeof(uint32_t)) != hipSuccess) {
    fprintf(stderr, "[tiled] allocation failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  memdbg(11, "bf_allocs");
  if (hipStreamCreateWithFlags(&s_work, hipStreamNonBlocking) != hipSuccess ||
      hipStreamCreateWithFlags(&s_copy, hipStreamNonBlocking) != hipSuccess ||
      hipEventCreateWithFlags(&ev_copy[0], hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&ev_copy[1], hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&ev_final, hipEventDisableTiming) != hipSuccess) {
    fprintf(stderr, "[tiled] stream/event setup failed\n");
    goto cleanup;
  }
  ev_ready = 1;
  _t0 = sk_now_ms();
  if (hipMemcpy(d_base, base, (size_t)n * dim * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
    fprintf(stderr, "[tiled] initial copy failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  {
    long pad_elems = (padded_n - n) * dim;
    if (pad_elems > 0) {
      fill_pad_kernel<<<(unsigned int)((pad_elems + 255) / 256), 256>>>(
          d_base, (long)n * dim, pad_elems, 1.0e6f);
    }
  }
  if (hipGetLastError() != hipSuccess ||
      hipMemset(d_norms, 0, (size_t)padded_n * sizeof(float)) != hipSuccess) {
    fprintf(stderr, "[tiled] initial copy failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  row_norm_kernel<<<(unsigned int)padded_n, 256>>>(d_base, d_norms, padded_n, dim);
  if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
    fprintf(stderr, "[tiled] norm kernel failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  if (g_time) g_t_copyin = sk_now_ms() - _t0;
  if (use_blas) {
    if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "[tiled] hipblas setup failed\n");
      goto cleanup;
    }
    hipblasSetPointerMode(blas, HIPBLAS_POINTER_MODE_HOST);
    hipblasSetStream(blas, s_work);
  }
  memdbg(12, "bf_gemm_ready");
  graph = (uint32_t*)sk_big_alloc((size_t)n * K * sizeof(uint32_t));
  if (!graph) { fprintf(stderr, "[tiled] host graph allocation failed\n"); goto cleanup; }
  if (hipHostMalloc((void**)&stage[0], (size_t)B * K * sizeof(uint32_t), 0) != hipSuccess ||
      hipHostMalloc((void**)&stage[1], (size_t)B * K * sizeof(uint32_t), 0) != hipSuccess) {
    fprintf(stderr, "[tiled] async copy setup failed\n"); goto cleanup;
  }

  for (long q0 = 0; q0 < padded_n; q0 += B) {
    if (q0 >= n) continue;
    int qrows = (int)((n - q0) < B ? (n - q0) : B);
    for (int tile = 0; tile < ntiles; ++tile) {
      long x0 = (long)tile * T;
      _t0 = sk_now_ms();
      if (use_blas) {
        const float alpha = -2.0f, beta = 0.0f;
        /* Full tile width keeps the hipBLAS kernel choice (and therefore the
           bit pattern of every distance) identical for every tile. */
        hipblasStatus_t hs = hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
            T, B, (int)dim, &alpha,
            d_base + (size_t)x0 * dim, (int)dim,
            d_base + (size_t)q0 * dim, (int)dim,
            &beta, d_dist, T);
        if (hs != HIPBLAS_STATUS_SUCCESS) {
          fprintf(stderr, "[tiled] sgemm failed q=%ld tile=%d status=%d\n", q0, tile, (int)hs);
          goto cleanup;
        }
        if (g_time) { hipStreamSynchronize(s_work); g_t_gemm += sk_now_ms() - _t0; g_n_gemm++; }
      } else {
        dim3 gdim((unsigned int)((B + DT_BM - 1) / DT_BM), (unsigned int)(T / DT_BN));
        dist_tile_kernel<<<gdim, 256, 0, s_work>>>(d_base, d_base, d_norms, d_dist,
                                        (int)dim, q0, x0, B, T, n);
      }
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] distance kernel failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      if (g_time && !use_blas) { hipStreamSynchronize(s_work); g_t_fin += sk_now_ms() - _t0; }
      _t0 = sk_now_ms();
      {
        int cv = (n - x0) < (long)T ? (int)(n - x0) : T;
        if (cv < 0) cv = 0;
        if (use_blas)
          select_tile_fused_kernel<true><<<(unsigned int)B, STK_NT, 0, s_work>>>(
              d_dist, d_norms, x0, cv, T, K, q0,
              cand_k + tile * K, cand_id + tile * K, cpr);
        else
          select_tile_fused_kernel<false><<<(unsigned int)B, STK_NT, 0, s_work>>>(
              d_dist, d_norms, x0, cv, T, K, q0,
              cand_k + tile * K, cand_id + tile * K, cpr);
      }
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] tile select failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      if (g_time) { hipStreamSynchronize(s_work); g_t_tsel += sk_now_ms() - _t0; }
    }

    /* Recycle the pinned staging buffer owned by the oldest pending copy. */
    if (copy_npend == 2) {
      int b0 = pend_buf[0];
      if (sk_finish_graph_copy(graph, K, stage[b0], ev_copy[b0], pend_q0[0], pend_rows[0]) != 0) {
        fprintf(stderr, "[tiled] copy wait failed\n"); goto cleanup;
      }
      pend_buf[0] = pend_buf[1]; pend_q0[0] = pend_q0[1]; pend_rows[0] = pend_rows[1];
      copy_npend = 1;
    }

    _t0 = sk_now_ms();
    {
      int gb = copy_gen & 1;
      if (copy_gen >= 2 && hipStreamWaitEvent(s_work, ev_copy[gb], 0) != hipSuccess) {
        fprintf(stderr, "[tiled] graph buffer wait failed\n"); goto cleanup;
      }
      select_topk_kernel<<<(unsigned int)B, 256, 0, s_work>>>(
          cand_k, cand_id, cpr, cpr, K, 0, K, d_graph_k, d_graph_block[gb]);
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] final select failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      if (g_time) { hipStreamSynchronize(s_work); g_t_fsel += sk_now_ms() - _t0; }
    }
    _t0 = sk_now_ms();
    {
      int gb = copy_gen & 1;
      if (hipEventRecord(ev_final, s_work) != hipSuccess ||
          hipStreamWaitEvent(s_copy, ev_final, 0) != hipSuccess) {
        fprintf(stderr, "[tiled] graph copy order failed\n"); goto cleanup;
      }
      if (hipMemcpyAsync(stage[gb], d_graph_block[gb],
                         (size_t)qrows * K * sizeof(uint32_t), hipMemcpyDeviceToHost, s_copy) != hipSuccess ||
          hipEventRecord(ev_copy[gb], s_copy) != hipSuccess) {
        fprintf(stderr, "[tiled] graph async copy failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      pend_buf[copy_npend] = gb; pend_q0[copy_npend] = q0; pend_rows[copy_npend] = qrows;
      copy_npend++;
      copy_gen++;
    }
    if (g_time) g_t_gcopy += sk_now_ms() - _t0;
  }

  while (copy_npend > 0) {
    int b0 = pend_buf[0];
    if (sk_finish_graph_copy(graph, K, stage[b0], ev_copy[b0], pend_q0[0], pend_rows[0]) != 0) {
      fprintf(stderr, "[tiled] final copy wait failed\n"); goto cleanup;
    }
    pend_buf[0] = pend_buf[1]; pend_q0[0] = pend_q0[1]; pend_rows[0] = pend_rows[1];
    copy_npend--;
  }
  memdbg(13, "bf_loop_done");
  if (hipDeviceSynchronize() != hipSuccess) goto cleanup;
  if (g_time) { g_t_total = sk_now_ms() - _t_fn; sk_time_report("bf_graph"); }
  *graph_out = graph;
  graph = NULL;
  rc = 0;

cleanup:
  memdbg(14, "bf_cleanup");
  if (blas) hipblasDestroy(blas);
  if (ev_ready) {
    hipEventDestroy(ev_copy[0]); hipEventDestroy(ev_copy[1]);
    hipEventDestroy(ev_final);
  }
  if (s_copy) hipStreamDestroy(s_copy);
  if (s_work) hipStreamDestroy(s_work);
  if (stage[0]) hipHostFree(stage[0]);
  if (stage[1]) hipHostFree(stage[1]);
  if (d_graph_block[0]) hipFree(d_graph_block[0]);
  if (d_graph_block[1]) hipFree(d_graph_block[1]);
  if (d_graph_k) hipFree(d_graph_k);
  if (cand_id) hipFree(cand_id);
  if (cand_k) hipFree(cand_k);
  if (d_dist) hipFree(d_dist);
  if (d_norms) hipFree(d_norms);
  if (d_base) hipFree(d_base);
  sk_big_free(graph, (size_t)n * (size_t)K * sizeof(uint32_t));
  memdbg(15, "bf_freed");
  return rc;
}

static int build_small_cagra(sk_bf_index* h, const float* base, long n, long dim)
{
  cuvsCagraIndexParams_t ip = NULL;
  DLManagedTensor base_t = {0};
  long bs[2] = {n, dim};
  long gd = n < 64 ? n : 64;
  if (cuvsCagraIndexParamsCreate(&ip) != CUVS_SUCCESS) { ip = NULL; goto fail; }
  ip->build_algo = AUTO_SELECT;
  ip->graph_degree = (size_t)gd;
  ip->intermediate_graph_degree = (size_t)gd;
  ip->nn_descent_niter = 20;
  set_tensor(&base_t, (void*)base, kDLCPU, 32, n, dim, bs);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraBuild(h->res, ip, &base_t, h->idx) != CUVS_SUCCESS) goto fail;
  cuvsCagraIndexParamsDestroy(ip);
  h->n = n;
  h->dim = dim;
  fprintf(stderr, "[tiled] small-n CAGRA build n=%ld dim=%ld gd=%ld\n", n, dim, gd);
  return 0;
fail:
  if (h->idx) { cuvsCagraIndexDestroy(h->idx); h->idx = NULL; }
  if (ip) cuvsCagraIndexParamsDestroy(ip);
  fprintf(stderr, "[tiled] small-n CAGRA build failed: %s\n", cuvsGetLastErrorText());
  return 1;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_bf_index* h = (sk_bf_index*)calloc(1, sizeof(*h));
  if (!h || !base || n <= 0 || dim <= 0) { free(h); return NULL; }
  memdbg(5, "build_entry");
  h->n = n; h->dim = dim;
  h->host_base = NULL;
  h->host_graph = NULL;
  h->graph_k = 0;
  h->index_k = 0;
  int rerank_l = env_int("TILED_RERANK_L", h->dim == 128 ? 0 : 256);
  int expand_seeds = env_int("TILED_EXPAND_SEEDS", 0);
  if (rerank_l > 0 || expand_seeds > 0) {
    size_t base_bytes = (size_t)n * (size_t)dim * sizeof(float);
    h->host_base = (float*)sk_big_alloc(base_bytes);
    if (!h->host_base) { free(h); return NULL; }
    memcpy(h->host_base, base, base_bytes);
  }
  mem_sampler_start();
  h->fp16 = env_int("TILED_FP16_INDEX", 0);
  h->k = (dim == 128) ? env_int("TILED_K", 64) : env_int("TILED_K", 256);
  int B = env_int("TILED_Q_BLOCK", dim == 128 ? 1024 : 768);
  int T = env_int("TILED_BASE_TILE", dim == 128 ? 8192 : 28672);
  if (B <= 0 || T <= 0 || h->k <= 0 || h->k > 256) { free(h); return NULL; }
  if (h->k > n) {
    long kk = 1;
    while (kk * 2 <= n && kk * 2 <= 256) kk *= 2;
    h->k = kk;
  }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h->host_base); free(h); return NULL; }
  memdbg(20, "res_created");

  /* GPU-path audit may build a tiny sample.  The tiled kernel pads to
     lcm(B,T); small n then produces invalid padding neighbors and can poison
     the GPU context.  Use cuVS CAGRA's own GPU builder for these tiny indexes. */
  uint32_t* graph = NULL;
  if (n < 4096) {
    if (build_small_cagra(h, base, n, dim) != 0) goto fail;
    return h;
  }

  if (build_exact_tiled_graph(base, n, dim, (int)h->k, B, T, &graph) != 0) goto fail;
  memdbg(21, "bf_graph_done");
  if (env_int("TILED_SKIP_INDEX", 0)) {
    free(graph);
    return h;
  }
  /* BF uses direct HIP allocations; do not reset the process-wide RMM pool:
     the seekdb service may have other CAGRA indexes alive in the same process. */
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { free(graph); goto fail; }
  memdbg(22, "import_begin");
  {
    DLManagedTensor graph_t = {0}, base_t = {0};
    long gs[2] = {n, h->k}, bs[2] = {n, dim};
    int index_cap = env_int("TILED_INDEX_K", 128);
    int index_k = h->k > index_cap ? index_cap : (int)h->k;
    uint32_t* compact_graph = NULL;
    uint32_t* index_graph = graph;
    if (index_k != h->k) {
      compact_graph = (uint32_t*)malloc((size_t)n * index_k * sizeof(uint32_t));
      if (!compact_graph) { free(graph); goto fail; }
      for (long row = 0; row < n; ++row)
        memcpy(compact_graph + (size_t)row * index_k,
               graph + (size_t)row * h->k,
               (size_t)index_k * sizeof(uint32_t));
      index_graph = compact_graph;
    }
    gs[1] = index_k;
    uint32_t* d_index_graph = NULL;
    if (env_int("TILED_DEVICE_GRAPH", 0)) {
      if (hipMalloc((void**)&d_index_graph, (size_t)n * index_k * sizeof(uint32_t)) != hipSuccess) {
        free(compact_graph); free(graph); goto fail;
      }
      if (hipMemcpy(d_index_graph, index_graph, (size_t)n * index_k * sizeof(uint32_t),
                    hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(d_index_graph); free(compact_graph); free(graph); goto fail;
      }
      set_tensor(&graph_t, d_index_graph, kDLCUDA, 32, n, index_k, gs);
    } else {
      set_tensor(&graph_t, index_graph, kDLCPU, 32, n, index_k, gs);
    }
    graph_t.dl_tensor.dtype.code = kDLUInt;
    unsigned short* half_base = NULL;
    void* pinned_base = NULL;
    if (!h->fp16 && env_int("TILED_PINNED_DATASET", 0)) {
      if (hipHostMalloc(&pinned_base, (size_t)n * (size_t)dim * sizeof(float), 0) == hipSuccess) {
        memcpy(pinned_base, base, (size_t)n * (size_t)dim * sizeof(float));
      } else {
        pinned_base = NULL;
      }
    }
    if (h->fp16) {
      half_base = (unsigned short*)malloc((size_t)n * (size_t)dim * sizeof(unsigned short));
      if (!half_base) { free(compact_graph); free(graph); goto fail; }
      for (size_t i = 0, cnt = (size_t)n * (size_t)dim; i < cnt; ++i) half_base[i] = f2h(base[i]);
      set_tensor(&base_t, (void*)half_base, kDLCPU, 16, n, dim, bs);
    } else {
      set_tensor(&base_t, pinned_base ? pinned_base : (void*)base, kDLCPU, 32, n, dim, bs);
    }
    memdbg(26, "import_src_ready");
    double _ti0 = sk_now_ms();
    if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_t, &base_t, h->idx) != CUVS_SUCCESS) {
      fprintf(stderr, "[tiled] CAGRA import failed: %s\n", cuvsGetLastErrorText());
      if (d_index_graph) hipFree(d_index_graph);
      if (pinned_base) hipHostFree(pinned_base);
      free(half_base); free(compact_graph); free(graph); goto fail;
    }
    if (g_time) {
      hipDeviceSynchronize();
      g_t_import = sk_now_ms() - _ti0;
      g_t_total += g_t_import;
      sk_time_report("bf_total");
    }
    if (d_index_graph) { h->d_graph = d_index_graph; }
    if (pinned_base) { h->pinned_base = pinned_base; }
    h->index_k = index_k;
    free(half_base);
    free(compact_graph);
    memdbg(23, "import_done");
    if (env_int("TILED_MEMDBG", 0)) {
      DLManagedTensor dv; memset(&dv, 0, sizeof(dv));
      if (cuvsCagraIndexGetDataset(h->idx, &dv) == CUVS_SUCCESS)
        fprintf(stderr, "[memdbg] dataset view dev=%d bits=%d ptr=%p\n",
                (int)dv.dl_tensor.device.device_type, (int)dv.dl_tensor.dtype.bits, dv.dl_tensor.data);
    }
  }
  if (expand_seeds > 0) {
    h->host_graph = graph;
    h->graph_k = h->k;
  } else {
    sk_big_free(graph, (size_t)n * (size_t)h->k * sizeof(uint32_t));
  }
  memdbg(24, "build_return");
  if (env_int("TILED_MEMDBG", 0) == 1) mem_sampler_stop();
  fprintf(stderr, "[tiled] built n=%ld dim=%ld k=%ld B=%d T=%d\n", n, dim, h->k, B, T);
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->res) cuvsResourcesDestroy(h->res);
  if (env_int("TILED_MEMDBG", 0) == 2) {
    mem_sampler_stop();
    fprintf(stderr, "[memdbg] fullrun peak=%.3f GB at phase=%d\n", (double)g_mem_peak.load() / 1e9, g_mem_peak_phase.load());
  }
  sk_big_free(h->host_base, (size_t)h->n * (size_t)h->dim * sizeof(float));
  free(h->host_graph);
  free(h->stamp);
  free(h);
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_bf_index* h = (sk_bf_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  long search_k = topk;
  int rerank_l = env_int("TILED_RERANK_L", h->dim == 128 ? 0 : 256);
  int expand_seeds = env_int("TILED_EXPAND_SEEDS", 0);
  int second_hop = env_int("TILED_SECOND_HOP", 0);
  int expand_deg = env_int("TILED_EXPAND_DEGREE", 0);
  if (expand_deg <= 0 || expand_deg > h->graph_k) expand_deg = (int)h->graph_k;
  int do_refine = h->host_base && (rerank_l > topk || expand_seeds > 0);
  if (do_refine && search_k < rerank_l) search_k = rerank_l;
  if (do_refine && search_k < expand_seeds) search_k = expand_seeds;
  float* dq = NULL; unsigned* dids = NULL; float* ddist = NULL;
  unsigned short* hq_half = NULL; unsigned short* dq_half = NULL;
  size_t qbytes = (size_t)nq * h->dim * 4;
  unsigned* h_candidates = NULL;
  struct Ranked* ranked = NULL;
  size_t candidate_cap = (size_t)search_k;
  if (do_refine && h->host_graph && expand_seeds > 0)
    candidate_cap += (size_t)(expand_seeds + (second_hop > 0 ? second_hop : 0)) * (size_t)expand_deg;
  cuvsCagraSearchParams_t sp = NULL;
  if (h->fp16) {
    hq_half = (unsigned short*)malloc((size_t)nq * (size_t)h->dim * sizeof(unsigned short));
    if (!hq_half) { rc = 10; goto done; }
    for (size_t i = 0, cnt = (size_t)nq * (size_t)h->dim; i < cnt; ++i) hq_half[i] = f2h(query[i]);
    if (cudaMalloc((void**)&dq_half, (size_t)nq * h->dim * 2) != cudaSuccess) { rc = 10; goto done; }
    if (cudaMemcpy(dq_half, hq_half, (size_t)nq * h->dim * 2, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  }
  if (!h->fp16 && cudaMalloc((void**)&dq, qbytes) != cudaSuccess) {
    fprintf(stderr, "[tiled] cudaMalloc dq(%zu) failed: %s\n", qbytes, cudaGetErrorString(cudaGetLastError()));
    rc = 10; goto done;
  }
  if (cudaMalloc((void**)&dids, (size_t)nq * search_k * 4) != cudaSuccess) { rc = 11; goto done; }
  if (cudaMalloc((void**)&ddist, (size_t)nq * search_k * 4) != cudaSuccess) { rc = 12; goto done; }
  if (!h->fp16 && cudaMemcpy(dq, query, qbytes, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  if (do_refine) {
    h_candidates = (unsigned*)malloc((size_t)nq * search_k * sizeof(unsigned));
    ranked = (struct Ranked*)malloc(candidate_cap * sizeof(struct Ranked));
    if (!h_candidates || !ranked) { rc = 18; goto done; }
  }
  {
    DLManagedTensor qt, nt, dt; long qs[2] = {nq, h->dim}, os[2] = {nq, search_k};
    set_tensor(&qt, h->fp16 ? (void*)dq_half : (void*)dq, kDLCUDA, h->fp16 ? 16 : 32, nq, h->dim, qs);
    set_tensor(&nt, dids, kDLCUDA, 32, nq, search_k, os); nt.dl_tensor.dtype.code = kDLUInt;
    set_tensor(&dt, ddist, kDLCUDA, 32, nq, search_k, os);
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 14; goto done; }
    size_t itopk = (size_t)env_int("BF_ITOPK", 512);
    if (itopk < (size_t)search_k) itopk = (size_t)search_k;
    if (sp->itopk_size < itopk) sp->itopk_size = itopk;
    sp->search_width = env_int("BF_WIDTH", 32);
    int hm = env_int("TILED_HASHMAP_MODE", -1);
    if (hm >= 0) sp->hashmap_mode = (enum cuvsCagraHashMode)hm;
    cuvsFilter filter; filter.type = NO_FILTER; filter.addr = (uintptr_t)0;
    memdbg(40, "cagra_search_begin");
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, filter) != CUVS_SUCCESS) { rc = 15; goto done; }
    memdbg(41, "cagra_search_end");
    if (do_refine) {
      if (cudaMemcpy(h_candidates, dids, (size_t)nq * search_k * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
      if (!h->stamp) {
        h->stamp = (uint32_t*)calloc((size_t)h->n, sizeof(uint32_t));
        if (!h->stamp) { rc = 18; goto done; }
        h->stamp_gen = 0;
      }
      long seeds = expand_seeds > 0 ? expand_seeds : 0;
      if (seeds > search_k) seeds = search_k;
      for (long qi = 0; qi < nq; ++qi) {
        uint32_t tag = ++h->stamp_gen;
        if (tag == 0) { memset(h->stamp, 0, (size_t)h->n * sizeof(uint32_t)); h->stamp_gen = 1; tag = 1; }
        const unsigned* seeds_row = h_candidates + (size_t)qi * search_k;
        size_t count = 0;
        for (long c = 0; c < search_k; ++c) {
          unsigned id = seeds_row[c];
          if ((long)id < h->n && h->stamp[id] != tag) { h->stamp[id] = tag; ranked[count].id = id; ranked[count].d = 0.0f; ++count; }
        }
        if (h->host_graph && expand_seeds > 0) {
          for (long c = 0; c < seeds; ++c) {
            unsigned seed = seeds_row[c];
            if ((long)seed >= h->n) continue;
            const uint32_t* row = h->host_graph + (size_t)seed * h->graph_k;
            for (int j = 0; j < expand_deg; ++j) {
              unsigned id = row[j];
              if ((long)id < h->n && h->stamp[id] != tag) { h->stamp[id] = tag; ranked[count].id = id; ranked[count].d = 0.0f; ++count; }
            }
          }
        }
        const float* q = query + (size_t)qi * h->dim;
        for (size_t c = 0; c < count; ++c)
          ranked[c].d = l2_dist(q, h->host_base + (size_t)ranked[c].id * h->dim, h->dim);
        if (h->host_graph && second_hop > 0) {
          size_t h2 = (size_t)second_hop < count ? (size_t)second_hop : count;
          std::nth_element(ranked, ranked + h2, ranked + count, ranked_less);
          size_t base_count = count;
          for (size_t c = 0; c < h2; ++c) {
            const uint32_t* row = h->host_graph + (size_t)ranked[c].id * h->graph_k;
            for (int j = 0; j < expand_deg; ++j) {
              unsigned id = row[j];
              if ((long)id < h->n && h->stamp[id] != tag) { h->stamp[id] = tag; ranked[count].id = id; ranked[count].d = 0.0f; ++count; }
            }
          }
          for (size_t c = base_count; c < count; ++c)
            ranked[c].d = l2_dist(q, h->host_base + (size_t)ranked[c].id * h->dim, h->dim);
        }
        size_t kk = (size_t)topk < count ? (size_t)topk : count;
        std::nth_element(ranked, ranked + kk, ranked + count, ranked_less);
        std::sort(ranked, ranked + kk, ranked_less);
        for (long j = 0; j < topk; ++j) {
          if ((size_t)j < kk) {
            out_ids[(size_t)qi * topk + j] = ranked[j].id;
            if (out_dist) out_dist[(size_t)qi * topk + j] = ranked[j].d;
          } else {
            out_ids[(size_t)qi * topk + j] = 0;
            if (out_dist) out_dist[(size_t)qi * topk + j] = INFINITY;
          }
        }
      }
    } else {
      if (cudaMemcpy(out_ids, dids, (size_t)nq * topk * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
      if (out_dist && cudaMemcpy(out_dist, ddist, (size_t)nq * topk * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 17; goto done; }
    }
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (ddist) cudaFree(ddist);
  if (dids) cudaFree(dids);
  if (dq) cudaFree(dq);
  if (dq_half) cudaFree(dq_half);
  free(hq_half);
  free(h_candidates); free(ranked);
  if (rc) fprintf(stderr, "[tiled] search rc=%d: %s\\n", rc, cuvsGetLastErrorText());
  return rc;
}
void seekdb_cuvs_free(void* handle)
{
  sk_bf_index* h = (sk_bf_index*)handle;
  if (!h) return;
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->d_graph) cudaFree(h->d_graph);
  if (h->d_dataset) cudaFree(h->d_dataset);
  if (h->pinned_base) hipHostFree(h->pinned_base);
  if (h->res) cuvsResourcesDestroy(h->res);
  if (env_int("TILED_MEMDBG", 0) == 2) {
    mem_sampler_stop();
    fprintf(stderr, "[memdbg] fullrun peak=%.3f GB at phase=%d\n", (double)g_mem_peak.load() / 1e9, g_mem_peak_phase.load());
  }
  sk_big_free(h->host_base, (size_t)h->n * (size_t)h->dim * sizeof(float));
  free(h->host_graph);
  free(h->stamp);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_bf_index* h = (sk_bf_index*)handle;
  if (!h || !filename) return 1;
  if (env_int("TILED_SKIP_SERIALIZE", 0)) return 0;
  memdbg(30, "serialize_begin");
  if (cuvsCagraSerialize(h->res, filename, h->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[bf_cagra] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  memdbg(31, "serialize_cuvs_done");
  if (h->host_graph && h->graph_k > 0) {
    int64_t hn = h->n, hd = h->dim, hk = h->graph_k;
    int64_t first = h->index_k;
    if (first < 0 || first >= hk) first = hk / 2;
    uint64_t data_bytes = (uint64_t)(5 * sizeof(int64_t)) +
                          (uint64_t)hn * (uint64_t)(hk - first) * sizeof(uint32_t);
    int embed = env_int("TILED_EMBED_SIDECAR", h->dim == 128 ? 0 : 1);
    if (embed) {
      FILE* f = fopen(filename, "ab");
      if (!f) { fprintf(stderr, "[tiled] embed open failed: %s\n", filename); return 0; }
      uint64_t magic = 0x544C4433ULL;
      int bad = 0;
      bad |= fwrite(&magic, sizeof(magic), 1, f) != 1;
      bad |= fwrite(&hn, sizeof(hn), 1, f) != 1;
      bad |= fwrite(&hd, sizeof(hd), 1, f) != 1;
      bad |= fwrite(&hk, sizeof(hk), 1, f) != 1;
      bad |= fwrite(&first, sizeof(first), 1, f) != 1;
      for (long row = 0; row < hn; ++row)
        bad |= fwrite(h->host_graph + (size_t)row * hk + first, sizeof(uint32_t),
                      (size_t)(hk - first), f) != (size_t)(hk - first);
      bad |= fwrite(&data_bytes, sizeof(data_bytes), 1, f) != 1;
      bad |= fwrite(&magic, sizeof(magic), 1, f) != 1;
      if (fclose(f) != 0) bad = 1;
      if (bad) { fprintf(stderr, "[tiled] compact sidecar write failed\n"); return 1; }
    }
  }
  memdbg(32, "serialize_done");
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_bf_index* h = (sk_bf_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  long long sc_start = -1;
  uint64_t sc_sz = 0;
  int64_t sc_hk = 0, sc_first = 0;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { fprintf(stderr, "[deser] resources failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] index create failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] cagra deserialize failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  h->fp16 = (h->idx && h->idx->dtype.bits == 16) ? 1 : 0;
  {
    FILE* f = fopen(filename, "rb");
    if (f) {
      if (fseek(f, 0, SEEK_END) == 0) {
        long long fs = ftell(f);
        if (fs > 48) {
          uint64_t sz = 0, tmagic = 0;
          if (fseek(f, (long)(fs - 16), SEEK_SET) == 0 &&
              fread(&sz, sizeof(sz), 1, f) == 1 &&
              fread(&tmagic, sizeof(tmagic), 1, f) == 1 &&
              tmagic == 0x544C4433ULL && sz > 0 && (long long)sz + 16 <= fs) {
            if (fseek(f, (long)(fs - 16 - (long long)sz), SEEK_SET) == 0) {
              uint64_t magic = 0; int64_t hn = 0, hd = 0, hk = 0, first = 0;
              if (fread(&magic, sizeof(magic), 1, f) == 1 &&
                  fread(&hn, sizeof(hn), 1, f) == 1 &&
                  fread(&hd, sizeof(hd), 1, f) == 1 &&
                  fread(&hk, sizeof(hk), 1, f) == 1 &&
                  fread(&first, sizeof(first), 1, f) == 1 &&
                  magic == 0x544C4433ULL && hn == n && hd == dim &&
                  hk > 0 && hk <= 256 && first >= 0 && first < hk) {
                sc_start = fs - 16 - (long long)sz;
                sc_sz = sz; sc_hk = hk; sc_first = first;
              }
            }
          }
        }
      }
      fclose(f);
    }
  }
  /* Recover the FP32 base from the CAGRA snapshot (no base sidecar). */
  {
    DLManagedTensor dv; memset(&dv, 0, sizeof(dv));
    if (cuvsCagraIndexGetDataset(h->idx, &dv) == CUVS_SUCCESS &&
        dv.dl_tensor.data && dv.dl_tensor.dtype.bits == 32 &&
        dv.dl_tensor.shape[0] == n && dv.dl_tensor.shape[1] == dim) {
      h->host_base = (float*)sk_big_alloc((size_t)n * dim * sizeof(float));
      if (h->host_base &&
          cudaMemcpy(h->host_base, dv.dl_tensor.data, (size_t)n * dim * sizeof(float),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        sk_big_free(h->host_base, (size_t)n * dim * sizeof(float)); h->host_base = NULL;
      }
    }
  }
  /* First graph degree comes from the CAGRA index; the rest from the compact sidecar. */
  {
    DLManagedTensor gv; memset(&gv, 0, sizeof(gv));
    if (cuvsCagraIndexGetGraph(h->idx, &gv) == CUVS_SUCCESS && gv.dl_tensor.data &&
        gv.dl_tensor.dtype.bits == 32 && gv.dl_tensor.shape[0] == n) {
      long gdeg = (long)gv.dl_tensor.shape[1];
      if (gdeg > 0 && gdeg <= 256) {
        long full = (sc_hk > gdeg) ? (long)sc_hk : gdeg;
        uint32_t* tmp = (uint32_t*)malloc((size_t)n * gdeg * sizeof(uint32_t));
        if (tmp && cudaMemcpy(tmp, gv.dl_tensor.data, (size_t)n * gdeg * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost) == cudaSuccess) {
          h->host_graph = (uint32_t*)malloc((size_t)n * full * sizeof(uint32_t));
          if (h->host_graph) {
            for (long row = 0; row < n; ++row)
              memcpy(h->host_graph + (size_t)row * full, tmp + (size_t)row * gdeg,
                     (size_t)gdeg * sizeof(uint32_t));
            h->graph_k = full;
            h->index_k = gdeg;
            if (sc_start >= 0 && sc_hk == full && sc_first == gdeg) {
              FILE* f = fopen(filename, "rb");
              if (f) {
                int ok = fseek(f, (long)(sc_start + 5 * (long long)sizeof(int64_t)), SEEK_SET) == 0;
                for (long row = 0; ok && row < n; ++row)
                  ok = fread(h->host_graph + (size_t)row * full + gdeg, sizeof(uint32_t),
                             (size_t)(full - gdeg), f) == (size_t)(full - gdeg);
                fclose(f);
                if (!ok) { free(h->host_graph); h->host_graph = NULL; h->graph_k = 0; h->index_k = 0; }
              }
            }
          }
        }
        free(tmp);
      }
    }
  }
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->res) cuvsResourcesDestroy(h->res);
  free(h);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  if (!base || !query || !out_ids || n <= 0 || dim <= 0 || nq <= 0 || topk <= 0) return 1;
  if (n < 64) {
    if (topk > n) return 1;
    for (long qi = 0; qi < nq; ++qi) {
      float best[64];
      unsigned int bid[64];
      for (long t = 0; t < topk; ++t) { best[t] = 3.4e38f; bid[t] = 0xffffffffu; }
      for (long j = 0; j < n; ++j) {
        float d = 0.0f;
        for (long k = 0; k < dim; ++k) {
          float x = base[j * dim + k] - query[qi * dim + k];
          d = fmaf(x, x, d);
        }
        long pos = topk;
        for (long t = 0; t < topk; ++t) {
          if (d < best[t]) { pos = t; break; }
        }
        if (pos < topk) {
          for (long t = topk - 1; t > pos; --t) { best[t] = best[t - 1]; bid[t] = bid[t - 1]; }
          best[pos] = d; bid[pos] = (unsigned int)j;
        }
      }
      for (long t = 0; t < topk; ++t) out_ids[qi * topk + t] = bid[t];
    }
    return 0;
  }
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
