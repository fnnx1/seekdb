/* seekdb bridge, HOST-ONLY build:
 *   exact top-K neighbor graph via the cuVS brute-force C API
 *   -> cuvsCagraIndexFromArgs -> CAGRA search + exact host rerank.
 *
 * This translation unit contains NO __global__ kernels and no device
 * intrinsics.  It is meant to be compiled by a plain host compiler
 * (clang++ / hipcc -x c++) without `-x hip` and without `--offload-arch`,
 * so the platform build does not pay device-code compile time.
 *
 * All GPU work is performed by hipVS / cuVS library entry points (which own
 * their device code); everything here is host bookkeeping.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <cuvs/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SK_HOST_X86 1
#endif

/* ------------------------------------------------------------------ */
/* timing / memory instrumentation (env gated: TILED_TIME, TILED_MEMDBG) */
static int g_time = 0;
static double g_t_half = 0, g_t_h2d = 0, g_t_bfbuild = 0, g_t_gsearch = 0,
              g_t_wait = 0, g_t_import = 0, g_t_total = 0;

static inline double sk_now_ms()
{
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int env_int(const char* key, int dflt)
{
  const char* v = getenv(key);
  return v && *v ? atoi(v) : dflt;
}

static std::atomic<int> g_mem_sampling{0};
static std::atomic<long long> g_mem_peak{0};
static std::atomic<int> g_mem_phase{0};
static std::atomic<int> g_mem_peak_phase{0};

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
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    g_mem_peak.store(best, std::memory_order_relaxed);
    g_mem_peak_phase.store(best_phase, std::memory_order_relaxed);
  }).detach();
}

static void mem_sampler_stop()
{
  if (!env_int("TILED_MEMDBG", 0)) return;
  g_mem_sampling.store(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
}

/* ------------------------------------------------------------------ */
/* index handle                                                        */
typedef struct {
  cuvsResources_t res;
  cuvsCagraIndex_t idx;
  long n, dim;
  long k;            /* graph degree imported into CAGRA */
  float* host_base;  /* n*dim fp32, for exact host rerank */
  uint32_t* stamp;   /* per-query duplicate suppression in rerank */
  uint32_t stamp_gen;
  hipStream_t s_search;  /* non-blocking stream handed to cuVS (pipeline) */
  hipStream_t s_copy;    /* D2H staging stream */
} sk_host_index;

/* ------------------------------------------------------------------ */
/* fp32 -> fp16 conversion (F16C when available, scalar otherwise)      */
static inline uint16_t f2h_scalar(float f)
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

#ifdef SK_HOST_X86
__attribute__((target("avx2,f16c")))
static void f2h_range_f16c(const float* src, uint16_t* dst, long count)
{
  long i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i*)(dst + i), h);
  }
  for (; i < count; ++i) dst[i] = f2h_scalar(src[i]);
}
static int g_have_f16c = -1;
#endif

static void f2h_all(const float* src, uint16_t* dst, long count)
{
  int nthreads = env_int("TILED_CONVERT_THREADS", 8);
  if (nthreads < 1) nthreads = 1;
  if (nthreads > 32) nthreads = 32;
  if (count < (1L << 20)) nthreads = 1;
  auto work = [&](long begin, long end) {
#ifdef SK_HOST_X86
    if (g_have_f16c < 0) g_have_f16c = __builtin_cpu_supports("f16c") ? 1 : 0;
    if (g_have_f16c) { f2h_range_f16c(src + begin, dst + begin, end - begin); return; }
#endif
    for (long i = begin; i < end; ++i) dst[i] = f2h_scalar(src[i]);
  };
  if (nthreads == 1) { work(0, count); return; }
  std::vector<std::thread> th;
  long per = (count + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    long b = (long)t * per;
    long e = b + per < count ? b + per : count;
    if (b >= count) break;
    th.emplace_back(work, b, e);
  }
  for (auto& x : th) x.join();
}

/* ------------------------------------------------------------------ */
static void set_tensor_f32(DLManagedTensor* t, void* data, DLDeviceType dev,
                           int bits, long r, long c, long* shape)
{
  memset(t, 0, sizeof(*t));
  shape[0] = r; shape[1] = c;
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = dev;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = kDLFloat;
  t->dl_tensor.dtype.bits = (uint8_t)bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = NULL;
}

static void set_tensor_i64(DLManagedTensor* t, void* data, DLDeviceType dev,
                           long r, long c, long* shape)
{
  set_tensor_f32(t, data, dev, 64, r, c, shape);
  t->dl_tensor.dtype.code = kDLInt;
}

/* ------------------------------------------------------------------ */
/* exact host L2 for rerank                                            */
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

#ifdef SK_HOST_X86
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
static inline float l2_dist(const float* q, const float* x, long dim)
{
  return l2_dist_scalar(q, x, dim);
}
#endif

/* ------------------------------------------------------------------ */
static int build_small_cagra(sk_host_index* h, const float* base, long n, long dim)
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
  set_tensor_f32(&base_t, (void*)base, kDLCPU, 32, n, dim, bs);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraBuild(h->res, ip, &base_t, h->idx) != CUVS_SUCCESS) goto fail;
  cuvsCagraIndexParamsDestroy(ip);
  h->n = n; h->dim = dim; h->k = gd;
  fprintf(stderr, "[host] small-n CAGRA build n=%ld dim=%ld gd=%ld\n", n, dim, gd);
  return 0;
fail:
  if (h->idx) { cuvsCagraIndexDestroy(h->idx); h->idx = NULL; }
  if (ip) cuvsCagraIndexParamsDestroy(ip);
  fprintf(stderr, "[host] small-n CAGRA build failed: %s\n", cuvsGetLastErrorText());
  return 1;
}

static int build_cagra_direct(sk_host_index* h, const float* base, long n,
                              long dim, long gd)
{
  cuvsCagraIndexParams_t ip = NULL;
  DLManagedTensor base_t;
  long bs[2] = {n, dim};
  if (cuvsCagraIndexParamsCreate(&ip) != CUVS_SUCCESS) { ip = NULL; goto fail; }
  ip->build_algo = AUTO_SELECT;
  ip->graph_degree = (size_t)gd;
  ip->intermediate_graph_degree = (size_t)gd;
  ip->nn_descent_niter = env_int("TILED_NND_NITER", 20);
  set_tensor_f32(&base_t, (void*)base, kDLCPU, 32, n, dim, bs);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  {
    double _t0 = sk_now_ms();
    if (cuvsCagraBuild(h->res, ip, &base_t, h->idx) != CUVS_SUCCESS) goto fail;
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) goto fail;
    g_t_import = sk_now_ms() - _t0;
  }
  cuvsCagraIndexParamsDestroy(ip);
  h->k = gd;
  fprintf(stderr, "[host] direct CAGRA build n=%ld dim=%ld gd=%ld\n", n, dim, gd);
  return 0;
fail:
  if (h->idx) { cuvsCagraIndexDestroy(h->idx); h->idx = NULL; }
  if (ip) cuvsCagraIndexParamsDestroy(ip);
  fprintf(stderr, "[host] direct CAGRA build failed: %s\n", cuvsGetLastErrorText());
  return 1;
}

/* ------------------------------------------------------------------ */
/* Build the exact top-K graph with the cuVS brute-force C API.
 * Returns a host n*K uint32 graph.                                    */
static int build_exact_graph_bf(sk_host_index* h, const float* base, long n,
                                long dim, long K, uint32_t** graph_out)
{
  int rc = 1;
  int chunk = env_int("TILED_BF_CHUNK", 256);
  int use_half = env_int("TILED_BF_HALF", 1);
  int depth = env_int("TILED_BF_DEPTH", 4);
  if (chunk < 16) chunk = 16;
  if (chunk > 8192) chunk = 8192;
  if (depth < 2) depth = 2;
  if (depth > 8) depth = 8;
  if (chunk > n) chunk = (int)n;

  cuvsBruteForceIndex_t bf = NULL;
  void* d_base = NULL;
  float* d_dist = NULL;
  int64_t* d_ids[8] = {0};
  void* stage[8] = {0};
  hipEvent_t ev[8] = {0};
  hipEvent_t ev_search[8] = {0};
  long pend_q0[8], pend_rows[8];
  int pend_valid[8] = {0};
  uint32_t* graph = NULL;
  unsigned short* half_base = NULL;
  size_t elem = use_half ? 2 : 4;
  size_t out_bytes = (size_t)chunk * (size_t)K * sizeof(int64_t);

  memdbg(10, "bf_enter");
  double t_all0 = sk_now_ms();
  double t_wait0 = 0;
  long nchunks = 0;
  cuvsFilter filter;
  if (cuvsBruteForceIndexCreate(&bf) != CUVS_SUCCESS) {
    fprintf(stderr, "[host] bf index create failed: %s\n", cuvsGetLastErrorText());
    goto cleanup;
  }
  graph = (uint32_t*)malloc((size_t)n * (size_t)K * sizeof(uint32_t));
  if (!graph) { fprintf(stderr, "[host] host graph alloc failed\n"); goto cleanup; }

  /* dataset on device (fp16 halves both footprint and scan bandwidth) */
  {
    double _t0 = sk_now_ms();
    if (use_half) {
      half_base = (unsigned short*)malloc((size_t)n * (size_t)dim * sizeof(unsigned short));
      if (!half_base) { fprintf(stderr, "[host] half staging alloc failed\n"); goto cleanup; }
      f2h_all(base, half_base, (long)((size_t)n * (size_t)dim));
      g_t_half = sk_now_ms() - _t0;
      _t0 = sk_now_ms();
    }
    if (hipMalloc(&d_base, (size_t)n * (size_t)dim * elem) != hipSuccess) {
      fprintf(stderr, "[host] d_base alloc failed\n"); goto cleanup;
    }
    if (hipMemcpy(d_base, use_half ? (void*)half_base : (void*)base,
                  (size_t)n * (size_t)dim * elem, hipMemcpyHostToDevice) != hipSuccess) {
      fprintf(stderr, "[host] d_base copy failed\n"); goto cleanup;
    }
    if (hipDeviceSynchronize() != hipSuccess) goto cleanup;
    g_t_h2d = sk_now_ms() - _t0;
    free(half_base); half_base = NULL;
  }
  memdbg(11, "bf_dataset_ready");

  {
    double _t0 = sk_now_ms();
    DLManagedTensor ds; long dsh[2];
    set_tensor_f32(&ds, d_base, kDLCUDA, use_half ? 16 : 32, n, dim, dsh);
    if (cuvsBruteForceBuild(h->res, &ds, L2Expanded, 0.0f, bf) != CUVS_SUCCESS) {
      fprintf(stderr, "[host] bf build failed: %s\n", cuvsGetLastErrorText());
      goto cleanup;
    }
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) goto cleanup;
    g_t_bfbuild = sk_now_ms() - _t0;
  }
  memdbg(12, "bf_index_built");

  if (hipMalloc((void**)&d_dist, (size_t)chunk * (size_t)K * sizeof(float)) != hipSuccess) {
    fprintf(stderr, "[host] d_dist alloc failed\n"); goto cleanup;
  }
  for (int d = 0; d < depth; ++d) {
    if (hipMalloc((void**)&d_ids[d], out_bytes) != hipSuccess ||
        hipHostMalloc(&stage[d], out_bytes, 0) != hipSuccess ||
        hipEventCreateWithFlags(&ev[d], hipEventDisableTiming) != hipSuccess ||
        hipEventCreateWithFlags(&ev_search[d], hipEventDisableTiming) != hipSuccess) {
      fprintf(stderr, "[host] pipeline alloc failed (depth=%d)\n", d);
      goto cleanup;
    }
  }
  memdbg(13, "bf_pipeline_ready");

  filter.type = NO_FILTER; filter.addr = (uintptr_t)0;
  for (long q0 = 0; q0 < n; q0 += chunk) {
    long nq = (n - q0) < chunk ? (n - q0) : chunk;
    int slot = (int)(nchunks % depth);
    if (pend_valid[slot]) {
      if (hipEventSynchronize(ev[slot]) != hipSuccess) { fprintf(stderr, "[host] event wait failed\n"); goto cleanup; }
      double w0 = sk_now_ms();
      const int64_t* s = (const int64_t*)stage[slot];
      uint32_t* g = graph + (size_t)pend_q0[slot] * (size_t)K;
      size_t cnt = (size_t)pend_rows[slot] * (size_t)K;
      for (size_t z = 0; z < cnt; ++z) g[z] = (uint32_t)s[z];
      t_wait0 += sk_now_ms() - w0;
      pend_valid[slot] = 0;
    }
    DLManagedTensor qt, nt, dt; long qsh[2], osh[2];
    set_tensor_f32(&qt, (void*)((char*)d_base + (size_t)q0 * (size_t)dim * elem),
                   kDLCUDA, use_half ? 16 : 32, nq, dim, qsh);
    set_tensor_i64(&nt, d_ids[slot], kDLCUDA, nq, K, osh);
    set_tensor_f32(&dt, d_dist, kDLCUDA, 32, nq, K, osh);
    double _s0 = sk_now_ms();
    if (cuvsBruteForceSearch(h->res, bf, &qt, &nt, &dt, filter) != CUVS_SUCCESS) {
      fprintf(stderr, "[host] bf search failed q0=%ld: %s\n", q0, cuvsGetLastErrorText());
      goto cleanup;
    }
    g_t_gsearch += sk_now_ms() - _s0;
    /* Async D2H on the staging stream, ordered after the search event on the
     * cuVS stream.  Both streams are non-blocking, so the search stream keeps
     * running while the copy drains. */
    if (h->s_search && h->s_copy) {
      if (hipEventRecord(ev_search[slot], h->s_search) != hipSuccess ||
          hipStreamWaitEvent(h->s_copy, ev_search[slot], 0) != hipSuccess ||
          hipMemcpyAsync(stage[slot], d_ids[slot], (size_t)nq * (size_t)K * sizeof(int64_t),
                         hipMemcpyDeviceToHost, h->s_copy) != hipSuccess ||
          hipEventRecord(ev[slot], h->s_copy) != hipSuccess) {
        fprintf(stderr, "[host] async d2h failed q0=%ld\n", q0);
        goto cleanup;
      }
    } else if (hipMemcpyAsync(stage[slot], d_ids[slot],
                              (size_t)nq * (size_t)K * sizeof(int64_t),
                              hipMemcpyDeviceToHost, 0) != hipSuccess ||
               hipEventRecord(ev[slot], 0) != hipSuccess) {
      fprintf(stderr, "[host] async d2h failed q0=%ld\n", q0);
      goto cleanup;
    }
    pend_valid[slot] = 1; pend_q0[slot] = q0; pend_rows[slot] = nq;
    ++nchunks;
    if ((nchunks & 255) == 0) {
      memdbg(14, "bf_search_progress");
      if (g_time) {
        double el = sk_now_ms() - t_all0;
        fprintf(stderr, "[time] bf progress %ld/%ld chunks %.0fms (%.1f ms/kq)\n",
                q0 + nq, n, el, el / ((double)(q0 + nq) / 1000.0));
      }
    }
  }
  for (int d = 0; d < depth; ++d) {
    if (!pend_valid[d]) continue;
    if (hipEventSynchronize(ev[d]) != hipSuccess) { fprintf(stderr, "[host] final event wait failed\n"); goto cleanup; }
    double w0 = sk_now_ms();
    const int64_t* s = (const int64_t*)stage[d];
    uint32_t* g = graph + (size_t)pend_q0[d] * (size_t)K;
    size_t cnt = (size_t)pend_rows[d] * (size_t)K;
    for (size_t z = 0; z < cnt; ++z) g[z] = (uint32_t)s[z];
    t_wait0 += sk_now_ms() - w0;
    pend_valid[d] = 0;
  }
  g_t_wait = t_wait0;
  memdbg(15, "bf_graph_done");

  if (env_int("TILED_SELFCHECK", 0)) {
    long self = 0, invalid = 0;
    for (long r = 0; r < n; ++r) {
      if ((long)graph[(size_t)r * K] == r) self++;
      if ((long)graph[(size_t)r * K] >= n) invalid++;
    }
    fprintf(stderr, "[host] graph selfcheck: self=%ld/%ld invalid=%ld/%ld\n", self, n, invalid, n);
  }

  *graph_out = graph;
  graph = NULL;
  rc = 0;
cleanup:
  for (int d = 0; d < depth; ++d) {
    if (ev_search[d]) hipEventDestroy(ev_search[d]);
    if (ev[d]) hipEventDestroy(ev[d]);
    if (stage[d]) hipHostFree(stage[d]);
    if (d_ids[d]) hipFree(d_ids[d]);
  }
  if (d_dist) hipFree(d_dist);
  if (bf) cuvsBruteForceIndexDestroy(bf);
  if (d_base) hipFree(d_base);
  free(half_base);
  free(graph);
  memdbg(16, "bf_freed");
  return rc;
}

/* ------------------------------------------------------------------ */
void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  if (!base || n <= 0 || dim <= 0) return NULL;
  sk_host_index* h = (sk_host_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  memdbg(5, "build_entry");
  g_time = env_int("TILED_TIME", 0);
  g_t_half = g_t_h2d = g_t_bfbuild = g_t_gsearch = g_t_wait = g_t_import = 0;
  mem_sampler_start();

  /* Host copy of the base: needed for exact rerank at query time and as the
   * fp32 dataset for the CAGRA snapshot (survives after build returns). */
  double _t_all = sk_now_ms();
  {
    double _t0 = sk_now_ms();
    h->host_base = (float*)malloc((size_t)n * (size_t)dim * sizeof(float));
    if (!h->host_base) { free(h); return NULL; }
    memcpy(h->host_base, base, (size_t)n * (size_t)dim * sizeof(float));
    if (g_time) fprintf(stderr, "[time] host_base copy %.1f ms\n", sk_now_ms() - _t0);
  }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) {
    fprintf(stderr, "[host] resources create failed: %s\n", cuvsGetLastErrorText());
    free(h->host_base); free(h); return NULL;
  }
  /* Dedicated non-blocking streams: cuVS work on s_search, D2H staging on
   * s_copy, so the next search never waits for the previous copy. */
  if (hipStreamCreateWithFlags(&h->s_search, hipStreamNonBlocking) == hipSuccess &&
      hipStreamCreateWithFlags(&h->s_copy, hipStreamNonBlocking) == hipSuccess) {
    if (cuvsStreamSet(h->res, h->s_search) != CUVS_SUCCESS) {
      fprintf(stderr, "[host] cuvsStreamSet failed, using default stream\n");
      hipStreamDestroy(h->s_search); h->s_search = NULL;
      hipStreamDestroy(h->s_copy); h->s_copy = NULL;
    }
  } else {
    if (h->s_search) { hipStreamDestroy(h->s_search); h->s_search = NULL; }
    if (h->s_copy) { hipStreamDestroy(h->s_copy); h->s_copy = NULL; }
  }
  memdbg(20, "res_created");

  if (n < 4096) {
    if (build_small_cagra(h, base, n, dim) != 0) goto fail;
    if (env_int("TILED_MEMDBG", 0) == 2) mem_sampler_stop();
    return h;
  }
  {
    const char* bm = getenv("TILED_BUILD_MODE");
    if (bm && strcmp(bm, "cagra") == 0) {
      long gd = env_int("TILED_INDEX_K", 128);
      if (gd > 256) gd = 256; if (gd > n) gd = n; if (gd < 16) gd = 16;
      if (build_cagra_direct(h, base, n, dim, gd) != 0) goto fail;
      g_t_total = sk_now_ms() - _t_all;
      if (g_time)
        fprintf(stderr, "[time] build(cagra) total=%.1f import=%.1f ms\n", g_t_total, g_t_import);
      if (env_int("TILED_MEMDBG", 0) == 2) {
        mem_sampler_stop();
        fprintf(stderr, "[memdbg] fullrun peak=%.3f GB at phase=%d\n",
                (double)g_mem_peak.load() / 1e9, g_mem_peak_phase.load());
      }
      fprintf(stderr, "[host] built(cagra) n=%ld dim=%ld k=%ld\n", n, dim, h->k);
      return h;
    }
  }

  {
    long K = env_int("TILED_BF_K", 128);
    if (K > 256) K = 256;
    if (K > n) K = n;
    if (K < 16) K = 16;
    uint32_t* graph = NULL;
    if (build_exact_graph_bf(h, base, n, dim, K, &graph) != 0) goto fail;
    memdbg(21, "bf_done");

    if (env_int("TILED_SKIP_INDEX", 0)) { free(graph); return h; }

    if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { free(graph); goto fail; }
    memdbg(22, "import_begin");
    {
      DLManagedTensor graph_t, base_t;
      long gs[2] = {n, K}, bs[2] = {n, dim};
      set_tensor_f32(&graph_t, graph, kDLCPU, 32, n, K, gs);
      graph_t.dl_tensor.dtype.code = kDLUInt;
      set_tensor_f32(&base_t, (void*)h->host_base, kDLCPU, 32, n, dim, bs);
      double _t0 = sk_now_ms();
      if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_t, &base_t, h->idx) != CUVS_SUCCESS) {
        fprintf(stderr, "[host] CAGRA import failed: %s\n", cuvsGetLastErrorText());
        free(graph); goto fail;
      }
      if (cuvsStreamSync(h->res) != CUVS_SUCCESS) { free(graph); goto fail; }
      g_t_import = sk_now_ms() - _t0;
    }
    h->k = K;
    free(graph);
    memdbg(23, "import_done");
  }

  g_t_total = sk_now_ms() - _t_all;
  if (g_time) {
    fprintf(stderr, "[time] build total=%.1f half=%.1f h2d=%.1f bfbuild=%.1f "
            "gsearch=%.1f wait=%.1f import=%.1f other=%.1f ms\n",
            g_t_total, g_t_half, g_t_h2d, g_t_bfbuild, g_t_gsearch, g_t_wait,
            g_t_import,
            g_t_total - (g_t_half + g_t_h2d + g_t_bfbuild + g_t_gsearch +
                         g_t_wait + g_t_import));
  }
  memdbg(24, "build_return");
  if (env_int("TILED_MEMDBG", 0) == 2) {
    mem_sampler_stop();
    fprintf(stderr, "[memdbg] fullrun peak=%.3f GB at phase=%d\n",
            (double)g_mem_peak.load() / 1e9, g_mem_peak_phase.load());
  }
  fprintf(stderr, "[host] built n=%ld dim=%ld k=%ld\n", n, dim, h->k);
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->res) cuvsResourcesDestroy(h->res);
  if (env_int("TILED_MEMDBG", 0) == 2) {
    mem_sampler_stop();
    fprintf(stderr, "[memdbg] fullrun peak=%.3f GB at phase=%d\n",
            (double)g_mem_peak.load() / 1e9, g_mem_peak_phase.load());
  }
  free(h->host_base);
  free(h->stamp);
  free(h);
  return NULL;
}

/* ------------------------------------------------------------------ */
int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_host_index* h = (sk_host_index*)handle;
  if (!h || !h->idx || !query || !out_ids || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  int rerank_l = env_int("TILED_RERANK_L", h->dim == 128 ? 0 : 256);
  if (rerank_l < topk) rerank_l = (int)topk;
  if (rerank_l > 512) rerank_l = 512;
  long search_k = rerank_l;
  if (search_k > h->n) search_k = h->n;
  int do_refine = h->host_base && (long)topk < search_k;

  float* dq = NULL;
  unsigned* dids = NULL;
  float* ddist = NULL;
  unsigned* h_candidates = NULL;
  struct Ranked* ranked = NULL;
  size_t qbytes = (size_t)nq * (size_t)h->dim * sizeof(float);
  cuvsCagraSearchParams_t sp = NULL;

  if (cudaMalloc((void**)&dq, qbytes) != cudaSuccess) { rc = 10; goto done; }
  if (cudaMalloc((void**)&dids, (size_t)nq * (size_t)search_k * sizeof(unsigned)) != cudaSuccess) { rc = 11; goto done; }
  if (cudaMalloc((void**)&ddist, (size_t)nq * (size_t)search_k * sizeof(float)) != cudaSuccess) { rc = 12; goto done; }
  if (cudaMemcpy(dq, query, qbytes, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  if (do_refine) {
    h_candidates = (unsigned*)malloc((size_t)nq * (size_t)search_k * sizeof(unsigned));
    ranked = (struct Ranked*)malloc((size_t)search_k * sizeof(struct Ranked));
    if (!h_candidates || !ranked) { rc = 18; goto done; }
  }
  {
    DLManagedTensor qt, nt, dt;
    long qs[2] = {nq, h->dim}, os[2] = {nq, search_k};
    set_tensor_f32(&qt, dq, kDLCUDA, 32, nq, h->dim, qs);
    set_tensor_f32(&nt, dids, kDLCUDA, 32, nq, search_k, os);
    nt.dl_tensor.dtype.code = kDLUInt;
    set_tensor_f32(&dt, ddist, kDLCUDA, 32, nq, search_k, os);
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
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) { rc = 15; goto done; }
    memdbg(41, "cagra_search_end");
    if (do_refine) {
      if (cudaMemcpy(h_candidates, dids, (size_t)nq * (size_t)search_k * sizeof(unsigned),
                     cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
      if (!h->stamp) {
        h->stamp = (uint32_t*)calloc((size_t)h->n, sizeof(uint32_t));
        if (!h->stamp) { rc = 18; goto done; }
        h->stamp_gen = 0;
      }
      for (long qi = 0; qi < nq; ++qi) {
        uint32_t tag = ++h->stamp_gen;
        if (tag == 0) { memset(h->stamp, 0, (size_t)h->n * sizeof(uint32_t)); h->stamp_gen = 1; tag = 1; }
        const unsigned* cand = h_candidates + (size_t)qi * search_k;
        size_t count = 0;
        for (long c = 0; c < search_k; ++c) {
          unsigned id = cand[c];
          if ((long)id < h->n && h->stamp[id] != tag) {
            h->stamp[id] = tag;
            ranked[count].id = id; ranked[count].d = 0.0f; ++count;
          }
        }
        const float* q = query + (size_t)qi * h->dim;
        for (size_t c = 0; c < count; ++c)
          ranked[c].d = l2_dist(q, h->host_base + (size_t)ranked[c].id * h->dim, h->dim);
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
      if (cudaMemcpy(out_ids, dids, (size_t)nq * (size_t)topk * sizeof(unsigned),
                     cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
      if (out_dist && cudaMemcpy(out_dist, ddist, (size_t)nq * (size_t)topk * sizeof(float),
                                 cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 17; goto done; }
    }
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (ddist) cudaFree(ddist);
  if (dids) cudaFree(dids);
  if (dq) cudaFree(dq);
  free(h_candidates);
  free(ranked);
  if (rc) fprintf(stderr, "[host] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

/* ------------------------------------------------------------------ */
void seekdb_cuvs_free(void* handle)
{
  sk_host_index* h = (sk_host_index*)handle;
  if (!h) return;
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->res) cuvsResourcesDestroy(h->res);
  if (h->s_search) hipStreamDestroy(h->s_search);
  if (h->s_copy) hipStreamDestroy(h->s_copy);
  free(h->host_base);
  free(h->stamp);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_host_index* h = (sk_host_index*)handle;
  if (!h || !h->idx || !filename) return 1;
  if (env_int("TILED_SKIP_SERIALIZE", 0)) return 0;
  memdbg(30, "serialize_begin");
  if (cuvsCagraSerialize(h->res, filename, h->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[host] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  memdbg(31, "serialize_done");
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_host_index* h = (sk_host_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { fprintf(stderr, "[deser] resources failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] index create failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] cagra deserialize failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  /* Recover the FP32 base from the CAGRA snapshot (dataset is embedded). */
  {
    DLManagedTensor dv; memset(&dv, 0, sizeof(dv));
    if (cuvsCagraIndexGetDataset(h->idx, &dv) == CUVS_SUCCESS) {
      if (!dv.dl_tensor.data || dv.dl_tensor.shape[0] != n || dv.dl_tensor.shape[1] != dim) {
        fprintf(stderr, "[deser] snapshot shape mismatch (want %ldx%ld)\n", n, dim);
        goto fail;
      }
    }
    if (cuvsCagraIndexGetDataset(h->idx, &dv) == CUVS_SUCCESS && dv.dl_tensor.data &&
        dv.dl_tensor.dtype.bits == 32 && dv.dl_tensor.shape[0] == n && dv.dl_tensor.shape[1] == dim) {
      h->host_base = (float*)malloc((size_t)n * (size_t)dim * sizeof(float));
      if (h->host_base &&
          cudaMemcpy(h->host_base, dv.dl_tensor.data, (size_t)n * (size_t)dim * sizeof(float),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(h->host_base); h->host_base = NULL;
      }
    }
  }
  {
    int64_t gd = 0;
    if (cuvsCagraIndexGetGraphDegree(h->idx, &gd) == CUVS_SUCCESS) h->k = (long)gd;
  }
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->res) cuvsResourcesDestroy(h->res);
  if (h->s_search) hipStreamDestroy(h->s_search);
  if (h->s_copy) hipStreamDestroy(h->s_copy);
  free(h->host_base);
  free(h);
  return NULL;
}

/* ------------------------------------------------------------------ */
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
        for (long t = 0; t < topk; ++t) if (d < best[t]) { pos = t; break; }
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
