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

typedef struct {
  cuvsResources_t res;
  cuvsCagraIndex_t idx;
  long n, dim, k;
  void* d_dataset;
  void* d_graph;
  float* host_base;
  uint32_t* host_graph;
  long graph_k;
  int fp16;
  uint32_t* stamp;
  uint32_t stamp_gen;
  void* pinned_base;
} sk_bf_index;

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
  uint32_t* d_graph_block = NULL;
  hipblasHandle_t blas = NULL;
  int use_blas = env_int("TILED_GEMM_BLAS", 0);
  uint32_t* graph = NULL;
  long lcm = (long)B * (long)T / gcd_int(B, T);
  long padded_n = ((n + lcm - 1) / lcm) * lcm;
  int ntiles = (int)(padded_n / T);
  int cpr = ntiles * K;
  size_t tile_elems = (size_t)B * T;
  size_t candidate_elems = (size_t)B * cpr;

  memdbg(10, "bf_enter");
  if (n <= 0 || dim <= 0 || K <= 0 || K > 256 || (K & (K - 1)) != 0 || B <= 0 || T <= 0 ||
      padded_n > INT_MAX || tile_elems > UINT_MAX || candidate_elems > UINT_MAX) return 1;
  if (hipMalloc((void**)&d_base, (size_t)padded_n * dim * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_norms, (size_t)padded_n * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_dist, tile_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&cand_k, candidate_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&cand_id, candidate_elems * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&d_graph_k, (size_t)B * K * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_graph_block, (size_t)B * K * sizeof(uint32_t)) != hipSuccess) {
    fprintf(stderr, "[tiled] allocation failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  memdbg(11, "bf_allocs");
  if (hipMemcpy(d_base, base, (size_t)n * dim * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemset(d_base + (size_t)n * dim, 0, (size_t)(padded_n - n) * dim * sizeof(float)) != hipSuccess ||
      hipMemset(d_norms, 0, (size_t)padded_n * sizeof(float)) != hipSuccess) {
    fprintf(stderr, "[tiled] initial copy failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  row_norm_kernel<<<(unsigned int)padded_n, 256>>>(d_base, d_norms, padded_n, dim);
  if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
    fprintf(stderr, "[tiled] norm kernel failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  if (use_blas) {
    if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "[tiled] hipblas setup failed\n");
      goto cleanup;
    }
    hipblasSetPointerMode(blas, HIPBLAS_POINTER_MODE_HOST);
    hipblasSetStream(blas, 0);
  }
  memdbg(12, "bf_gemm_ready");
  graph = (uint32_t*)malloc((size_t)n * K * sizeof(uint32_t));
  if (!graph) { fprintf(stderr, "[tiled] host graph allocation failed\n"); goto cleanup; }

  for (long q0 = 0; q0 < padded_n; q0 += B) {
    if (q0 >= n) continue;
    int qrows = (int)((n - q0) < B ? (n - q0) : B);
    for (int tile = 0; tile < ntiles; ++tile) {
      long x0 = (long)tile * T;
      if (use_blas) {
        const float alpha = -2.0f, beta = 0.0f;
        hipblasStatus_t hs = hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
            T, B, (int)dim, &alpha,
            d_base + (size_t)x0 * dim, (int)dim,
            d_base + (size_t)q0 * dim, (int)dim,
            &beta, d_dist, T);
        if (hs != HIPBLAS_STATUS_SUCCESS) {
          fprintf(stderr, "[tiled] sgemm failed q=%ld tile=%d status=%d\n", q0, tile, (int)hs);
          goto cleanup;
        }
        finish_distance_tile<<<(unsigned int)((tile_elems + 255) / 256), 256>>>(
            d_dist, d_norms, n, q0, x0, B, T);
      } else {
        dim3 gdim((unsigned int)((B + DT_BM - 1) / DT_BM), (unsigned int)(T / DT_BN));
        dist_tile_kernel<<<gdim, 256>>>(d_base, d_base, d_norms, d_dist,
                                        (int)dim, q0, x0, B, T, n);
      }
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] distance kernel failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      select_topk_kernel<<<(unsigned int)B, 256>>>(
          d_dist, nullptr, T, T, K, x0, cpr, cand_k + tile * K, cand_id + tile * K);
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] tile select failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
    }
    select_topk_kernel<<<(unsigned int)B, 256>>>(
        cand_k, cand_id, cpr, cpr, K, 0, K, d_graph_k, d_graph_block);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
      fprintf(stderr, "[tiled] final select failed: %s\n", hipGetErrorString(hipGetLastError()));
      goto cleanup;
    }
    if (hipMemcpy(graph + (size_t)q0 * K, d_graph_block,
                  (size_t)qrows * K * sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
      fprintf(stderr, "[tiled] graph copy failed: %s\n", hipGetErrorString(hipGetLastError()));
      goto cleanup;
    }
  }

  memdbg(13, "bf_loop_done");
  if (hipDeviceSynchronize() != hipSuccess) goto cleanup;
  *graph_out = graph;
  graph = NULL;
  rc = 0;

cleanup:
  memdbg(14, "bf_cleanup");
  if (blas) hipblasDestroy(blas);
  if (d_graph_block) hipFree(d_graph_block);
  if (d_graph_k) hipFree(d_graph_k);
  if (cand_id) hipFree(cand_id);
  if (cand_k) hipFree(cand_k);
  if (d_dist) hipFree(d_dist);
  if (d_norms) hipFree(d_norms);
  if (d_base) hipFree(d_base);
  free(graph);
  memdbg(15, "bf_freed");
  return rc;
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
  int rerank_l = env_int("TILED_RERANK_L", 0);
  int expand_seeds = env_int("TILED_EXPAND_SEEDS", 0);
  if (rerank_l > 0 || expand_seeds > 0) {
    size_t base_bytes = (size_t)n * (size_t)dim * sizeof(float);
    h->host_base = (float*)malloc(base_bytes);
    if (!h->host_base) { free(h); return NULL; }
    memcpy(h->host_base, base, base_bytes);
  }
  mem_sampler_start();
  h->fp16 = env_int("TILED_FP16_INDEX", 0);
  h->k = (dim == 128) ? env_int("TILED_K", 64) : env_int("TILED_K", 256);
  int B = env_int("TILED_Q_BLOCK", dim == 128 ? 1024 : 256);
  int T = env_int("TILED_BASE_TILE", dim == 128 ? 8192 : 16384);
  if (B <= 0 || T <= 0 || h->k <= 0 || h->k > 256) { free(h); return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h->host_base); free(h); return NULL; }
  memdbg(20, "res_created");

  uint32_t* graph = NULL;
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
    if (env_int("TILED_DEVICE_GRAPH", dim == 128 ? 0 : 1)) {
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
    if (!h->fp16 && env_int("TILED_PINNED_DATASET", dim == 128 ? 0 : 1)) {
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
    if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_t, &base_t, h->idx) != CUVS_SUCCESS) {
      fprintf(stderr, "[tiled] CAGRA import failed: %s\n", cuvsGetLastErrorText());
      if (d_index_graph) hipFree(d_index_graph);
      if (pinned_base) hipHostFree(pinned_base);
      free(half_base); free(compact_graph); free(graph); goto fail;
    }
    if (d_index_graph) { h->d_graph = d_index_graph; }
    if (pinned_base) { h->pinned_base = pinned_base; }
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
    free(graph);
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
  free(h->host_base);
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
  int rerank_l = env_int("TILED_RERANK_L", 0);
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
  free(h->host_base);
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
  if (h->host_base && h->host_graph && h->graph_k > 0) {
    int64_t hn = h->n, hd = h->dim, hk = h->graph_k;
    uint64_t data_bytes = (uint64_t)(sizeof(uint64_t) + 3 * sizeof(int64_t)) +
                          (uint64_t)hn * hd * sizeof(float) +
                          (uint64_t)hn * hk * sizeof(uint32_t);
    int embed = env_int("TILED_EMBED_SIDECAR", 0);
    if (embed) {
      FILE* f = fopen(filename, "ab");
      if (!f) { fprintf(stderr, "[tiled] embed open failed: %s\n", filename); return 0; }
      uint64_t magic = 0x544C4432ULL;
      fwrite(&magic, sizeof(magic), 1, f);
      fwrite(&hn, sizeof(hn), 1, f);
      fwrite(&hd, sizeof(hd), 1, f);
      fwrite(&hk, sizeof(hk), 1, f);
      fwrite(h->host_base, sizeof(float), (size_t)hn * hd, f);
      fwrite(h->host_graph, sizeof(uint32_t), (size_t)hn * hk, f);
      fwrite(&data_bytes, sizeof(data_bytes), 1, f);
      fwrite(&magic, sizeof(magic), 1, f);
      fclose(f);
    } else {
      char path[4096];
      snprintf(path, sizeof(path), "%s.tiled", filename);
      FILE* f = fopen(path, "wb");
      if (!f) { fprintf(stderr, "[tiled] sidecar open failed: %s\n", path); return 0; }
      uint64_t magic = 0x544C4431ULL;
      fwrite(&magic, sizeof(magic), 1, f);
      fwrite(&hn, sizeof(hn), 1, f);
      fwrite(&hd, sizeof(hd), 1, f);
      fwrite(&hk, sizeof(hk), 1, f);
      fwrite(h->host_base, sizeof(float), (size_t)hn * hd, f);
      fwrite(h->host_graph, sizeof(uint32_t), (size_t)hn * hk, f);
      fclose(f);
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
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { fprintf(stderr, "[deser] resources failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] index create failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] cagra deserialize failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  h->fp16 = (h->idx && h->idx->dtype.bits == 16) ? 1 : 0;
  {
    FILE* f = fopen(filename, "rb");
    if (f) {
      int loaded = 0;
      if (!loaded && fseek(f, 0, SEEK_END) == 0) {
        long long fs = ftell(f);
        if (fs > 32) {
          uint64_t sz = 0, tmagic = 0;
          if (fseek(f, (long)(fs - 16), SEEK_SET) == 0 &&
              fread(&sz, sizeof(sz), 1, f) == 1 &&
              fread(&tmagic, sizeof(tmagic), 1, f) == 1 &&
              tmagic == 0x544C4432ULL && sz > 0 && (long long)sz + 16 <= fs) {
            if (fseek(f, (long)(fs - 16 - (long long)sz), SEEK_SET) == 0) {
              uint64_t magic = 0; int64_t hn = 0, hd = 0, hk = 0;
              if (fread(&magic, sizeof(magic), 1, f) == 1 &&
                  fread(&hn, sizeof(hn), 1, f) == 1 &&
                  fread(&hd, sizeof(hd), 1, f) == 1 &&
                  fread(&hk, sizeof(hk), 1, f) == 1 &&
                  magic == 0x544C4432ULL && hn == n && hd == dim && hk > 0 && hk <= 256) {
                h->host_base = (float*)malloc((size_t)hn * hd * sizeof(float));
                h->host_graph = (uint32_t*)malloc((size_t)hn * hk * sizeof(uint32_t));
                if (h->host_base && h->host_graph &&
                    fread(h->host_base, sizeof(float), (size_t)hn * hd, f) == (size_t)hn * hd &&
                    fread(h->host_graph, sizeof(uint32_t), (size_t)hn * hk, f) == (size_t)hn * hk) {
                  h->graph_k = hk;
                  loaded = 1;
                } else {
                  free(h->host_base); free(h->host_graph);
                  h->host_base = NULL; h->host_graph = NULL; h->graph_k = 0;
                }
              }
            }
          }
        }
      }
      if (!loaded) {
        char path[4096];
        snprintf(path, sizeof(path), "%s.tiled", filename);
        FILE* g = fopen(path, "rb");
        if (g) {
          uint64_t magic = 0; int64_t hn = 0, hd = 0, hk = 0;
          if (fread(&magic, sizeof(magic), 1, g) == 1 &&
              fread(&hn, sizeof(hn), 1, g) == 1 &&
              fread(&hd, sizeof(hd), 1, g) == 1 &&
              fread(&hk, sizeof(hk), 1, g) == 1 &&
              magic == 0x544C4431ULL && hn == n && hd == dim && hk > 0 && hk <= 256) {
            h->host_base = (float*)malloc((size_t)hn * hd * sizeof(float));
            h->host_graph = (uint32_t*)malloc((size_t)hn * hk * sizeof(uint32_t));
            if (h->host_base && h->host_graph &&
                fread(h->host_base, sizeof(float), (size_t)hn * hd, g) == (size_t)hn * hd &&
                fread(h->host_graph, sizeof(uint32_t), (size_t)hn * hk, g) == (size_t)hn * hk) {
              h->graph_k = hk;
            } else {
              free(h->host_base); free(h->host_graph);
              h->host_base = NULL; h->host_graph = NULL; h->graph_k = 0;
            }
          }
          fclose(g);
        }
      }
      fclose(f);
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
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
