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
#include <mutex>
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

static long long mem_used_bytes();
/* Fine-grained VRAM trace (TILED_VRAM_TRACE=1): samples every ~250 us and
 * records every new high-water mark together with the current step label. */
static std::atomic<int> g_vt_run{0};
static std::atomic<long long> g_vt_peak{0};
static std::atomic<int> g_vt_peak_step{0};
static std::atomic<double> g_vt_peak_ms{0};
static std::atomic<double> g_vt_t0{0};
static std::atomic<int> g_vt_step{0};
static const char* const g_vt_names[] = {
  "0", "1", "2", "3", "4", "5-build_entry", "6", "7", "8", "9",
  "10-bf_enter", "11-bf_dataset", "12-bf_index", "13-bf_pipeline", "14-bf_search",
  "15-bf_graph_done", "16-bf_freed", "17", "18", "19",
  "20-res_created", "21-bf_done", "22-import_begin", "23-import_done", "24-build_return",
  "25", "26", "27", "28", "29",
  "30-ser_begin", "31-ser_graph_d2h", "32-ser_data_d2h", "33-ser_write", "34-ser_done",
  "35", "36", "37", "38", "39",
  "40-search_begin", "41-search_done", "42-search_dq", "43-search_params", "44-search_cagra",
  "45-search_cagra_done", "46-search_d2h", "47-search_rerank", "48-search_free", "49",
  "50-snap_header", "51-snap_read_graph", "52-snap_read_data", "53-snap_res", "54-snap_fromargs",
  "55-snap_done", "56-snap_free", "57", "58", "59",
  "60-free_begin", "61-free_done"};
static inline void vt_step(int step)
{
  g_vt_step.store(step, std::memory_order_relaxed);
  g_mem_phase.store(step, std::memory_order_relaxed);
}
static void vt_start()
{
  if (!env_int("TILED_VRAM_TRACE", 0)) return;
  g_vt_t0.store(sk_now_ms());
  g_vt_run.store(1);
  std::thread([]() {
    long long best = 0;
    while (g_vt_run.load(std::memory_order_relaxed)) {
      long long u = mem_used_bytes();
      if (u > best) {
        best = u;
        int st = g_vt_step.load(std::memory_order_relaxed);
        g_vt_peak.store(best, std::memory_order_relaxed);
        g_vt_peak_step.store(st, std::memory_order_relaxed);
        g_vt_peak_ms.store(sk_now_ms() - g_vt_t0.load(), std::memory_order_relaxed);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(250));
    }
  }).detach();
}
static void vt_report(const char* tag)
{
  if (!env_int("TILED_VRAM_TRACE", 0)) return;
  int st = g_vt_peak_step.load();
  fprintf(stderr, "[vtrace] %-18s peak=%.3f GB step=%d(%s) at %.1f ms\n", tag,
          (double)g_vt_peak.load() / 1e9, st,
          (st >= 0 && st < (int)(sizeof(g_vt_names) / sizeof(g_vt_names[0]))) ? g_vt_names[st] : "?",
          g_vt_peak_ms.load());
}
static void vt_stop()
{
  if (!env_int("TILED_VRAM_TRACE", 0)) return;
  g_vt_run.store(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  int st = g_vt_peak_step.load();
  fprintf(stderr, "[vtrace] peak=%.3f GB step=%d(%s) at %.1f ms\n",
          (double)g_vt_peak.load() / 1e9, st,
          (st >= 0 && st < (int)(sizeof(g_vt_names) / sizeof(g_vt_names[0]))) ? g_vt_names[st] : "?",
          g_vt_peak_ms.load());
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
  int fp16;          /* CAGRA stores fp16 dataset/queries (halves VRAM+snapshot) */
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

#ifdef SK_HOST_X86
__attribute__((target("avx2,f16c")))
static void h2f_range_f16c(const uint16_t* src, float* dst, long count)
{
  long i = 0;
  for (; i + 8 <= count; i += 8) {
    __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
  }
  for (; i < count; ++i) {
    uint32_t h = src[i];
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t f;
    if (expo == 0) {
      if (mant == 0) f = sign;
      else { int e = -1; do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
             f = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13); }
    } else if (expo == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((expo + 112u) << 23) | (mant << 13);
    memcpy(dst + i, &f, 4);
  }
}
static int g_have_f16c_h2f = -1;
#endif

static void h2f_all(const uint16_t* src, float* dst, long count)
{
#ifdef SK_HOST_X86
  if (g_have_f16c_h2f < 0) g_have_f16c_h2f = __builtin_cpu_supports("f16c") ? 1 : 0;
  if (g_have_f16c_h2f) { h2f_range_f16c(src, dst, count); return; }
#endif
  for (long i = 0; i < count; ++i) {
    uint32_t h = src[i];
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1Fu, mant = h & 0x3FFu;
    uint32_t f;
    if (expo == 0) {
      if (mant == 0) f = sign;
      else { int e = -1; do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
             f = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13); }
    } else if (expo == 31) f = sign | 0x7F800000u | (mant << 13);
    else f = sign | ((expo + 112u) << 23) | (mant << 13);
    memcpy(dst + i, &f, 4);
  }
}

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
  int chunk = env_int("TILED_BF_CHUNK", 64);
  int use_half = env_int("TILED_BF_HALF", 1);
  int depth = env_int("TILED_BF_DEPTH", 2);
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
  vt_step(10);
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
    vt_step(11);
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
  vt_step(12);

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
  vt_step(13);

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
    vt_step(14);
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
  vt_step(15);

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
  vt_step(16);
  return rc;
}

/* Single-slot resources cache: resources destroyed per handle make the driver
 * keep ~180 MB of pools/workspaces alive after seekdb_cuvs_free(), inflating
 * the peak of a build -> free -> load -> query lifecycle.  The most recently
 * retired (resources, streams) pair is kept and handed to the next handle;
 * live handles never share resources, so queries stay independent. */
static std::mutex g_cache_mu;
static cuvsResources_t g_cache_res = NULL;
static hipStream_t g_cache_search = NULL;
static hipStream_t g_cache_copy = NULL;

static int sk_res_create(sk_host_index* h)
{
  if (env_int("TILED_RES_CACHE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    if (g_cache_res) {
      h->res = g_cache_res; h->s_search = g_cache_search; h->s_copy = g_cache_copy;
      g_cache_res = NULL; g_cache_search = NULL; g_cache_copy = NULL;
      return 0;
    }
  }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) return 1;
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
  return 0;
}

static void sk_res_release(sk_host_index* h)
{
  if (!h->res) return;
  if (env_int("TILED_RES_CACHE", 0)) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    if (!g_cache_res) {
      g_cache_res = h->res; g_cache_search = h->s_search; g_cache_copy = h->s_copy;
      h->res = NULL; h->s_search = NULL; h->s_copy = NULL;
      return;
    }
  }
  cuvsResourcesDestroy(h->res);
  if (h->s_search) hipStreamDestroy(h->s_search);
  if (h->s_copy) hipStreamDestroy(h->s_copy);
  h->res = NULL; h->s_search = NULL; h->s_copy = NULL;
}

/* ------------------------------------------------------------------ */
/* Custom snapshot (fp16 index): header + uint32 graph + fp16 dataset.
 * hipVS 26.03 cannot deserialize an fp16 CAGRA file, so this format is
 * rebuilt through cuvsCagraIndexFromArgs instead.                      */
#define SK_SNAP_MAGIC 0x534B4831u  /* "SKH1" */

static int snap_write(sk_host_index* h, const char* filename)
{
  DLManagedTensor gv, dv;
  memset(&gv, 0, sizeof(gv)); memset(&dv, 0, sizeof(dv));
  if (cuvsCagraIndexGetGraph(h->idx, &gv) != CUVS_SUCCESS || !gv.dl_tensor.data) {
    fprintf(stderr, "[host] snapshot: graph view failed\n"); return 1;
  }
  if (cuvsCagraIndexGetDataset(h->idx, &dv) != CUVS_SUCCESS || !dv.dl_tensor.data) {
    fprintf(stderr, "[host] snapshot: dataset view failed\n"); return 1;
  }
  int64_t n = dv.dl_tensor.shape[0], dim = dv.dl_tensor.shape[1];
  int64_t gdeg = gv.dl_tensor.shape[1];
  int bits = dv.dl_tensor.dtype.bits;
  if (n != h->n || dim != h->dim || gdeg <= 0 || gdeg > 256 || (bits != 16 && bits != 32)) {
    fprintf(stderr, "[host] snapshot: unexpected shape/dtype\n"); return 1;
  }
  size_t graph_bytes = (size_t)n * (size_t)gdeg * sizeof(uint32_t);
  size_t data_bytes = (size_t)n * (size_t)dim * (size_t)(bits / 8);
  uint32_t* hg = (uint32_t*)malloc(graph_bytes);
  void* hd = malloc(data_bytes);
  int rc = 1;
  if (!hg || !hd) { fprintf(stderr, "[host] snapshot: host alloc failed\n"); goto out; }
  if (cudaMemcpy(hg, gv.dl_tensor.data, graph_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(hd, dv.dl_tensor.data, data_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "[host] snapshot: d2h failed\n"); goto out;
  }
  {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "[host] snapshot: open %s failed\n", filename); goto out; }
    uint32_t magic = SK_SNAP_MAGIC, version = 1, u32bits = (uint32_t)bits;
    int64_t hn = n, hd_dim = dim, hk = gdeg;
    int bad = 0;
    bad |= fwrite(&magic, 4, 1, f) != 1;
    bad |= fwrite(&version, 4, 1, f) != 1;
    bad |= fwrite(&hn, 8, 1, f) != 1;
    bad |= fwrite(&hd_dim, 8, 1, f) != 1;
    bad |= fwrite(&hk, 8, 1, f) != 1;
    bad |= fwrite(&u32bits, 4, 1, f) != 1;
    bad |= fwrite(hg, 1, graph_bytes, f) != graph_bytes;
    bad |= fwrite(hd, 1, data_bytes, f) != data_bytes;
    if (fclose(f) != 0) bad = 1;
    if (bad) { fprintf(stderr, "[host] snapshot: write failed\n"); goto out; }
  }
  rc = 0;
out:
  free(hg); free(hd);
  return rc;
}

static void* snap_read(const char* filename, long n, long dim)
{
  FILE* f = fopen(filename, "rb");
  if (!f) return NULL;
  uint32_t magic = 0, version = 0, u32bits = 0;
  int64_t hn = 0, hdim = 0, hk = 0;
  int ok = fread(&magic, 4, 1, f) == 1 && fread(&version, 4, 1, f) == 1 &&
           fread(&hn, 8, 1, f) == 1 && fread(&hdim, 8, 1, f) == 1 &&
           fread(&hk, 8, 1, f) == 1 && fread(&u32bits, 4, 1, f) == 1;
  if (!ok || magic != SK_SNAP_MAGIC || version != 1 || hn != n || hdim != dim ||
      hk <= 0 || hk > 256 || (u32bits != 16 && u32bits != 32)) {
    fclose(f);
    fprintf(stderr, "[deser] custom snapshot header mismatch\n");
    return NULL;
  }
  size_t graph_bytes = (size_t)n * (size_t)hk * 4;
  size_t data_bytes = (size_t)n * (size_t)dim * (u32bits / 8);
  uint32_t* hg = (uint32_t*)malloc(graph_bytes);
  void* hd = malloc(data_bytes);
  sk_host_index* h = NULL;
  if (!hg || !hd) { free(hg); free(hd); fclose(f); return NULL; }
  if (fread(hg, 1, graph_bytes, f) != graph_bytes ||
      fread(hd, 1, data_bytes, f) != data_bytes) {
    fprintf(stderr, "[deser] custom snapshot short read\n");
    free(hg); free(hd); fclose(f); return NULL;
  }
  fclose(f);

  h = (sk_host_index*)calloc(1, sizeof(*h));
  if (!h) { free(hg); free(hd); return NULL; }
  h->n = n; h->dim = dim; h->k = (long)hk; h->fp16 = (u32bits == 16);
  if (sk_res_create(h) != 0) { fprintf(stderr, "[deser] resources failed: %s\n", cuvsGetLastErrorText()); goto fail; }
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { fprintf(stderr, "[deser] index create failed\n"); goto fail; }
  {
    DLManagedTensor gt, dt;
    long gs[2] = {n, (long)hk}, ds[2] = {n, dim};
    set_tensor_f32(&gt, hg, kDLCPU, 32, n, (long)hk, gs);
    gt.dl_tensor.dtype.code = kDLUInt;
    set_tensor_f32(&dt, hd, kDLCPU, (int)u32bits, n, dim, ds);
    vt_step(54);
    if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &gt, &dt, h->idx) != CUVS_SUCCESS) {
      fprintf(stderr, "[deser] CAGRA import failed: %s\n", cuvsGetLastErrorText()); goto fail;
    }
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) goto fail;
  }
  h->host_base = (float*)malloc((size_t)n * (size_t)dim * sizeof(float));
  if (!h->host_base) goto fail;
  if (h->fp16) h2f_all((const uint16_t*)hd, h->host_base, (long)((size_t)n * (size_t)dim));
  else memcpy(h->host_base, hd, (size_t)n * (size_t)dim * sizeof(float));
  free(hg); free(hd);
  vt_step(55);
  vt_report("after_deserialize");
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  sk_res_release(h);
  free(h->host_base); free(h);
  free(hg); free(hd);
  return NULL;
}

/* ------------------------------------------------------------------ */
void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  if (!base || n <= 0 || dim <= 0) return NULL;
  sk_host_index* h = (sk_host_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  memdbg(5, "build_entry");
  vt_start();
  vt_step(5);
  g_time = env_int("TILED_TIME", 0);
  g_t_half = g_t_h2d = g_t_bfbuild = g_t_gsearch = g_t_wait = g_t_import = 0;
  mem_sampler_start();

  /* Host copy of the base: needed for exact rerank at query time and as the
   * fp32 dataset for the CAGRA snapshot (survives after build returns).
   * Copied on a CPU thread while the GPU builds the graph. */
  double _t_all = sk_now_ms();
  std::thread copy_thread;
  {
    vt_step(6);
    h->host_base = (float*)malloc((size_t)n * (size_t)dim * sizeof(float));
    if (!h->host_base) { free(h); return NULL; }
    const float* src = base;
    float* dst = h->host_base;
    size_t bytes = (size_t)n * (size_t)dim * sizeof(float);
    copy_thread = std::thread([src, dst, bytes]() { memcpy(dst, src, bytes); });
  }
  auto join_copy = [&]() {
    if (copy_thread.joinable()) copy_thread.join();
  };
  /* Dedicated non-blocking streams: cuVS work on s_search, D2H staging on
   * s_copy, so the next search never waits for the previous copy. */
  if (sk_res_create(h) != 0) {
    fprintf(stderr, "[host] resources create failed: %s\n", cuvsGetLastErrorText());
    free(h->host_base); free(h); return NULL;
  }
  memdbg(20, "res_created");

  if (n < 4096) {
    if (build_small_cagra(h, base, n, dim) != 0) { join_copy(); goto fail; }
    join_copy();
    if (env_int("TILED_MEMDBG", 0) == 2) mem_sampler_stop();
    return h;
  }
  {
    const char* bm = getenv("TILED_BUILD_MODE");
    if (bm && strcmp(bm, "cagra") == 0) {
      long gd = env_int("TILED_INDEX_K", 128);
      if (gd > 256) gd = 256; if (gd > n) gd = n; if (gd < 16) gd = 16;
      if (build_cagra_direct(h, base, n, dim, gd) != 0) { join_copy(); goto fail; }
      join_copy();
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
    if (build_exact_graph_bf(h, base, n, dim, K, &graph) != 0) { join_copy(); goto fail; }
    join_copy();
    memdbg(21, "bf_done");

    if (env_int("TILED_SKIP_INDEX", 0)) { free(graph); return h; }

    if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { free(graph); goto fail; }
    memdbg(22, "import_begin");
    vt_step(22);
    {
      DLManagedTensor graph_t, base_t;
      long gs[2] = {n, K}, bs[2] = {n, dim};
      unsigned short* half_dataset = NULL;
      int fp16_index = env_int("TILED_FP16_INDEX", 1);
      set_tensor_f32(&graph_t, graph, kDLCPU, 32, n, K, gs);
      graph_t.dl_tensor.dtype.code = kDLUInt;
      if (fp16_index) {
        half_dataset = (unsigned short*)malloc((size_t)n * (size_t)dim * sizeof(unsigned short));
        if (!half_dataset) { free(graph); goto fail; }
        f2h_all(h->host_base, half_dataset, (long)((size_t)n * (size_t)dim));
        set_tensor_f32(&base_t, half_dataset, kDLCPU, 16, n, dim, bs);
      } else {
        set_tensor_f32(&base_t, (void*)h->host_base, kDLCPU, 32, n, dim, bs);
      }
      double _t0 = sk_now_ms();
      if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_t, &base_t, h->idx) != CUVS_SUCCESS) {
        fprintf(stderr, "[host] CAGRA import failed: %s\n", cuvsGetLastErrorText());
        free(half_dataset); free(graph); goto fail;
      }
      if (cuvsStreamSync(h->res) != CUVS_SUCCESS) { free(half_dataset); free(graph); goto fail; }
      g_t_import = sk_now_ms() - _t0;
      free(half_dataset);
      h->fp16 = fp16_index;
    }
    h->k = K;
    free(graph);
    memdbg(23, "import_done");
    vt_step(23);
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
  vt_step(24);
  vt_report("after_build");
  return h;
fail:
  vt_report("build_fail");
  join_copy();
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  sk_res_release(h);
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
  /* Optional query chunking: CAGRA search workspace grows ~1 MB/query, so
   * capping the batch bounds transient VRAM at a small latency cost. */
  {
    int qc = env_int("TILED_QUERY_CHUNK", 50);
    if (qc > 0 && nq > qc) {
      for (long off = 0; off < nq; off += qc) {
        long m = (nq - off) < qc ? (nq - off) : qc;
        int r = seekdb_cuvs_search(handle, query + (size_t)off * h->dim, m, topk,
                                   out_ids + (size_t)off * topk,
                                   out_dist ? out_dist + (size_t)off * topk : NULL);
        if (r) return r;
      }
      return 0;
    }
  }
  int rc = 0;
  int rerank_l = env_int("TILED_RERANK_L", h->dim == 128 ? 0 : 256);
  if (rerank_l < topk) rerank_l = (int)topk;
  if (rerank_l > 512) rerank_l = 512;
  long search_k = rerank_l;
  if (search_k > h->n) search_k = h->n;
  int do_refine = h->host_base && (long)topk < search_k;

  float* dq = NULL;
  unsigned short* dq_half = NULL;
  unsigned short* hq_half = NULL;
  unsigned* dids = NULL;
  float* ddist = NULL;
  unsigned* h_candidates = NULL;
  struct Ranked* ranked = NULL;
  size_t qbytes = (size_t)nq * (size_t)h->dim * sizeof(float);
  size_t qbytes_half = (size_t)nq * (size_t)h->dim * sizeof(unsigned short);
  cuvsCagraSearchParams_t sp = NULL;
  double t_search = 0, t_d2h = 0, t_rerank = 0;
  double _t0 = sk_now_ms();
  vt_step(42);

  if (h->fp16) {
    hq_half = (unsigned short*)malloc(qbytes_half);
    if (!hq_half) { rc = 10; goto done; }
    f2h_all(query, hq_half, (long)((size_t)nq * (size_t)h->dim));
    if (cudaMalloc((void**)&dq_half, qbytes_half) != cudaSuccess) { rc = 10; goto done; }
    if (cudaMemcpy(dq_half, hq_half, qbytes_half, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  } else {
    if (cudaMalloc((void**)&dq, qbytes) != cudaSuccess) { rc = 10; goto done; }
    if (cudaMemcpy(dq, query, qbytes, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  }
  memdbg(42, "search_after_dq");
  if (cudaMalloc((void**)&dids, (size_t)nq * (size_t)search_k * sizeof(unsigned)) != cudaSuccess) { rc = 11; goto done; }
  if (cudaMalloc((void**)&ddist, (size_t)nq * (size_t)search_k * sizeof(float)) != cudaSuccess) { rc = 12; goto done; }
  if (do_refine) {
    h_candidates = (unsigned*)malloc((size_t)nq * (size_t)search_k * sizeof(unsigned));
    ranked = (struct Ranked*)malloc((size_t)search_k * sizeof(struct Ranked));
    if (!h_candidates || !ranked) { rc = 18; goto done; }
  }
  {
    DLManagedTensor qt, nt, dt;
    long qs[2] = {nq, h->dim}, os[2] = {nq, search_k};
    set_tensor_f32(&qt, h->fp16 ? (void*)dq_half : (void*)dq, kDLCUDA,
                   h->fp16 ? 16 : 32, nq, h->dim, qs);
    set_tensor_f32(&nt, dids, kDLCUDA, 32, nq, search_k, os);
    nt.dl_tensor.dtype.code = kDLUInt;
    set_tensor_f32(&dt, ddist, kDLCUDA, 32, nq, search_k, os);
    memdbg(43, "search_before_params");
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 14; goto done; }
    size_t itopk = (size_t)env_int("BF_ITOPK", 512);
    if (itopk < (size_t)search_k) itopk = (size_t)search_k;
    if (sp->itopk_size < itopk) sp->itopk_size = itopk;
    sp->search_width = env_int("BF_WIDTH", 32);
    int hm = env_int("TILED_HASHMAP_MODE", -1);
    if (hm >= 0) sp->hashmap_mode = (enum cuvsCagraHashMode)hm;
    int alg = env_int("TILED_SEARCH_ALGO", -1);
    if (alg >= 0) sp->algo = (enum cuvsCagraSearchAlgo)alg;
    int tbs = env_int("TILED_THREAD_BLOCK", 0);
    if (tbs > 0) sp->thread_block_size = (size_t)tbs;
    int hmin = env_int("TILED_HASH_MIN_BITS", 0);
    if (hmin > 0) sp->hashmap_min_bitlen = (size_t)hmin;
    cuvsFilter filter; filter.type = NO_FILTER; filter.addr = (uintptr_t)0;
    memdbg(40, "cagra_search_begin");
    vt_step(44);
    double _ts = sk_now_ms();
    memdbg(44, "search_before_cagra");
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, filter) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) { rc = 15; goto done; }
    t_search = sk_now_ms() - _ts;
    memdbg(45, "cagra_search_end");
    vt_step(45);
    memdbg(41, "cagra_search_end2");
    if (do_refine) {
      vt_step(46);
      double _td = sk_now_ms();
      if (cudaMemcpy(h_candidates, dids, (size_t)nq * (size_t)search_k * sizeof(unsigned),
                     cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
      t_d2h = sk_now_ms() - _td;
      vt_step(47);
      double _tr = sk_now_ms();
      /* Deduplicate by sorting candidate ids (thread-local), then exact
       * fp32 AVX2 rerank; queries are independent so fan them out. */
      {
        int nth = env_int("TILED_RERANK_THREADS", 8);
        if (nth < 1) nth = 1;
        if ((long)nth > nq) nth = (int)nq;
        auto worker = [&](long qb, long qe) {
          std::vector<unsigned> cand((size_t)search_k);
          std::vector<Ranked> rank((size_t)search_k);
          for (long qi = qb; qi < qe; ++qi) {
            const unsigned* src = h_candidates + (size_t)qi * search_k;
            memcpy(cand.data(), src, (size_t)search_k * sizeof(unsigned));
            std::sort(cand.begin(), cand.end());
            size_t count = (size_t)(std::unique(cand.begin(), cand.end()) - cand.begin());
            const float* q = query + (size_t)qi * h->dim;
            for (size_t c = 0; c < count; ++c) {
              rank[c].id = cand[c];
              rank[c].d = l2_dist(q, h->host_base + (size_t)cand[c] * h->dim, h->dim);
            }
            size_t kk = (size_t)topk < count ? (size_t)topk : count;
            std::nth_element(rank.begin(), rank.begin() + kk, rank.begin() + count, ranked_less);
            std::sort(rank.begin(), rank.begin() + kk, ranked_less);
            for (long j = 0; j < topk; ++j) {
              if ((size_t)j < kk) {
                out_ids[(size_t)qi * topk + j] = rank[j].id;
                if (out_dist) out_dist[(size_t)qi * topk + j] = rank[j].d;
              } else {
                out_ids[(size_t)qi * topk + j] = 0;
                if (out_dist) out_dist[(size_t)qi * topk + j] = INFINITY;
              }
            }
          }
        };
        if (nth == 1) {
          worker(0, nq);
        } else {
          std::vector<std::thread> th;
          long per = (nq + nth - 1) / nth;
          for (int t = 0; t < nth; ++t) {
            long b = (long)t * per;
            long e = b + per < nq ? b + per : nq;
            if (b >= nq) break;
            th.emplace_back(worker, b, e);
          }
          for (auto& x : th) x.join();
        }
      }
      t_rerank = sk_now_ms() - _tr;
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
  if (dq_half) cudaFree(dq_half);
  vt_step(48);
  free(h_candidates);
  free(ranked);
  free(hq_half);
  if (g_time) {
    fprintf(stderr, "[time] search nq=%ld topk=%ld sk=%ld cagra=%.1f d2h=%.1f rerank=%.1f other=%.1f ms\n",
            nq, topk, search_k, t_search, t_d2h, t_rerank,
            (sk_now_ms() - _t0) - (t_search + t_d2h + t_rerank));
  }
  vt_report("after_search");
  if (rc) fprintf(stderr, "[host] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

/* ------------------------------------------------------------------ */
void seekdb_cuvs_free(void* handle)
{
  sk_host_index* h = (sk_host_index*)handle;
  if (!h) return;
  vt_step(60);
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  sk_res_release(h);
  free(h->host_base);
  free(h->stamp);
  free(h);
  /* Benchmark/eval convenience: the process-wide RMM pool keeps index and
   * search-workspace memory cached across free+reload, which inflates the
   * observed peak of a build->serialize->load->query cycle run in one
   * process.  Releasing it is only safe when this bridge owns every index in
   * the process, so it is opt-in. */
  if (env_int("TILED_RMM_RESET_ON_FREE", 0)) cuvsRMMMemoryResourceReset();
  vt_step(61);
  vt_report("after_free");
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_host_index* h = (sk_host_index*)handle;
  if (!h || !h->idx || !filename) return 1;
  if (env_int("TILED_SKIP_SERIALIZE", 0)) return 0;
  memdbg(30, "serialize_begin");
  vt_step(30);
  double _t0 = sk_now_ms();
  int rc;
  if (h->fp16) {
    rc = snap_write(h, filename);
  } else if (cuvsCagraSerialize(h->res, filename, h->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[host] serialize failed: %s\n", cuvsGetLastErrorText());
    rc = 1;
  } else {
    rc = 0;
  }
  if (g_time) fprintf(stderr, "[time] serialize %.1f ms\n", sk_now_ms() - _t0);
  memdbg(31, "serialize_done");
  vt_step(34);
  vt_report("after_serialize");
  return rc;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  {
    FILE* probe = fopen(filename, "rb");
    if (probe) {
      uint32_t magic = 0;
      size_t got = fread(&magic, 4, 1, probe);
      fclose(probe);
      if (got == 1 && magic == SK_SNAP_MAGIC) return snap_read(filename, n, dim);
    }
  }
  sk_host_index* h = (sk_host_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  if (sk_res_create(h) != 0) { fprintf(stderr, "[deser] resources failed: %s\n", cuvsGetLastErrorText()); goto fail; }
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
      h->fp16 = (dv.dl_tensor.dtype.bits == 16) ? 1 : 0;
      if (h->fp16) {
        unsigned short* half = (unsigned short*)malloc((size_t)n * (size_t)dim * sizeof(unsigned short));
        h->host_base = (float*)malloc((size_t)n * (size_t)dim * sizeof(float));
        if (half && h->host_base &&
            cudaMemcpy(half, dv.dl_tensor.data, (size_t)n * (size_t)dim * sizeof(unsigned short),
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
          h2f_all(half, h->host_base, (long)((size_t)n * (size_t)dim));
        } else {
          free(h->host_base); h->host_base = NULL;
        }
        free(half);
      } else {
        h->host_base = (float*)malloc((size_t)n * (size_t)dim * sizeof(float));
        if (h->host_base &&
            cudaMemcpy(h->host_base, dv.dl_tensor.data, (size_t)n * (size_t)dim * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          free(h->host_base); h->host_base = NULL;
        }
      }
    }
  }
  {
    int64_t gd = 0;
    if (cuvsCagraIndexGetGraphDegree(h->idx, &gd) == CUVS_SUCCESS) h->k = (long)gd;
  }
  vt_report("after_deserialize");
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  sk_res_release(h);
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
