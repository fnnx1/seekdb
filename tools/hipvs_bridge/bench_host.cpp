/* End-to-end bench for the host-only seekdb bridge.
 * Build -> search (R@10/R@100) -> serialize -> reload -> search again.
 * Per-phase VRAM peaks sampled via hipMemGetInfo.
 *
 * BENCH_ONLY=build   : build + build-index search only (fresh process).
 * BENCH_ONLY=reload  : deserialize + search only (fresh process).
 * BENCH_SKIP_SER=1   : build + search, no persistence round.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

static double now_ms()
{
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::atomic<int> g_sampling{0};
static std::atomic<long long> g_peak{0};
static std::atomic<long long> g_win_peak{0};
static std::atomic<int> g_peak_phase{0};
static std::atomic<int> g_phase{0};

static long long mem_used()
{
  size_t f = 0, tot = 0;
  if (hipMemGetInfo(&f, &tot) != hipSuccess) return -1;
  return (long long)(tot - f);
}
static void sampler_start()
{
  g_sampling.store(1);
  std::thread([]() {
    long long best = 0; int bp = 0;
    while (g_sampling.load()) {
      long long u = mem_used();
      if (u > best) { best = u; bp = g_phase.load(); }
      if (u > g_win_peak.load()) g_win_peak.store(u);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    g_peak.store(best); g_peak_phase.store(bp);
  }).detach();
}
static void sampler_stop() { g_sampling.store(0); std::this_thread::sleep_for(std::chrono::milliseconds(25)); }
static void win_reset() { g_win_peak.store(mem_used()); }
static double win_peak_gb() { return (double)g_win_peak.load() / 1e9; }
static double used_gb() { return (double)mem_used() / 1e9; }

static int read_file(const char* path, void* dst, size_t bytes)
{
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
  size_t got = fread(dst, 1, bytes, f);
  fclose(f);
  if (got != bytes) { fprintf(stderr, "read %s got %zu want %zu\n", path, got, bytes); return 1; }
  return 0;
}

static double recall_at(const uint32_t* ids, const int32_t* ex, int topk, int kq, int rk)
{
  long hit = 0;
  for (int i = 0; i < kq; ++i) {
    for (int j = 0; j < rk; ++j) {
      uint32_t id = ids[(size_t)i * topk + j];
      for (int t = 0; t < 100; ++t)
        if (ex[(size_t)i * 100 + t] == (int32_t)id) { ++hit; break; }
    }
  }
  return (double)hit / ((double)kq * rk);
}

int main(int argc, char** argv)
{
  const char* ddir = argc > 1 ? argv[1] : "/workspace/validate/data";
  const char* idxpath = argc > 2 ? argv[2] : "/workspace/validate/forkB/host/bench.idx";
  long n = 250000, dim = 256;
  int nq = getenv("BENCH_NQ") ? atoi(getenv("BENCH_NQ")) : 100;
  if (nq > 100) nq = 100;
  const int topk = 100;
  const char* only = getenv("BENCH_ONLY");
  int only_build = only && strcmp(only, "build") == 0;
  int only_reload = only && strcmp(only, "reload") == 0;
  int skip_ser = (getenv("BENCH_SKIP_SER") && atoi(getenv("BENCH_SKIP_SER"))) || only_build || only_reload;

  std::vector<float> base((size_t)n * dim);
  std::vector<float> q((size_t)nq * dim);
  std::vector<int32_t> ex((size_t)nq * 100);
  char p[1024];
  snprintf(p, sizeof(p), "%s/rbase.bin", ddir);
  if (read_file(p, base.data(), base.size() * 4)) return 1;
  snprintf(p, sizeof(p), "%s/rq.bin", ddir);
  if (read_file(p, q.data(), q.size() * 4)) return 1;
  snprintf(p, sizeof(p), "%s/rex.bin", ddir);
  if (read_file(p, ex.data(), ex.size() * 4)) return 1;

  printf("[bench] n=%ld dim=%ld nq=%d topk=%d\n", n, dim, nq, topk);
  fflush(stdout);
  sampler_start();

  double t_build = 0, t_search = 0, t_ser = 0, t_deser = 0, t_search2 = 0;
  std::vector<uint32_t> ids((size_t)nq * topk), ids2((size_t)nq * topk);
  std::vector<float> dist((size_t)nq * topk), dist2((size_t)nq * topk);
  void* h = NULL;

  if (!only_reload) {
    g_phase.store(1);
    win_reset();
    double t0 = now_ms();
    h = seekdb_cuvs_build(base.data(), n, dim);
    t_build = now_ms() - t0;
    if (!h) { fprintf(stderr, "build failed\n"); return 2; }
    printf("[bench] build: %.1f ms   [vram] used=%.3f GB build_peak=%.3f GB\n",
           t_build, used_gb(), win_peak_gb());
    fflush(stdout);

    g_phase.store(2);
    win_reset();
    double s0 = now_ms();
    int rc = seekdb_cuvs_search(h, q.data(), nq, topk, ids.data(), dist.data());
    t_search = now_ms() - s0;
    if (rc) { fprintf(stderr, "search failed rc=%d\n", rc); return 3; }
    printf("[bench] build-index search (topk=%d): %.1f ms  R@10=%.4f R@100=%.4f  "
           "[vram] used=%.3f GB search_peak=%.3f GB\n",
           topk, t_search, recall_at(ids.data(), ex.data(), topk, nq, 10),
           recall_at(ids.data(), ex.data(), topk, nq, 100), used_gb(), win_peak_gb());
    fflush(stdout);
  }

  if (!skip_ser) {
    g_phase.store(3);
    win_reset();
    double z0 = now_ms();
    if (seekdb_cuvs_serialize(h, idxpath) != 0) { fprintf(stderr, "serialize failed\n"); return 4; }
    t_ser = now_ms() - z0;
    {
      FILE* f = fopen(idxpath, "rb");
      long long sz = 0;
      if (f) { fseek(f, 0, SEEK_END); sz = ftell(f); fclose(f); }
      printf("[bench] serialize: %.1f ms  idx=%.1f MB  [vram] used=%.3f GB ser_peak=%.3f GB\n",
             t_ser, (double)sz / 1e6, used_gb(), win_peak_gb());
      fflush(stdout);
    }
    seekdb_cuvs_free(h);
    h = NULL;
    printf("[bench] after free: used=%.3f GB\n", used_gb());
    fflush(stdout);
  }

  if (only_build) {
    sampler_stop();
    printf("[bench] VRAM full-run peak=%.3f GB (phase=%d)\n",
           (double)g_peak.load() / 1e9, g_peak_phase.load());
    printf("[bench] SUMMARY build=%.1fms search=%.1fms R@10=%.4f R@100=%.4f\n",
           t_build, t_search, recall_at(ids.data(), ex.data(), topk, nq, 10),
           recall_at(ids.data(), ex.data(), topk, nq, 100));
    return 0;
  }

  {
    g_phase.store(4);
    win_reset();
    double d0 = now_ms();
    h = seekdb_cuvs_deserialize(idxpath, n, dim);
    t_deser = now_ms() - d0;
    if (!h) { fprintf(stderr, "deserialize failed\n"); return 5; }
    printf("[bench] deserialize: %.1f ms   [vram] used=%.3f GB deser_peak=%.3f GB\n",
           t_deser, used_gb(), win_peak_gb());
    fflush(stdout);
  }

  {
    g_phase.store(5);
    win_reset();
    double s1 = now_ms();
    int rc = seekdb_cuvs_search(h, q.data(), nq, topk, ids2.data(), dist2.data());
    t_search2 = now_ms() - s1;
    if (rc) { fprintf(stderr, "post-reload search failed rc=%d\n", rc); return 6; }
    printf("[bench] post-reload search: %.1f ms  R@10=%.4f R@100=%.4f  "
           "[vram] used=%.3f GB reload_peak=%.3f GB\n", t_search2,
           recall_at(ids2.data(), ex.data(), topk, nq, 10),
           recall_at(ids2.data(), ex.data(), topk, nq, 100), used_gb(), win_peak_gb());
    fflush(stdout);
  }

  sampler_stop();
  printf("[bench] VRAM full-run peak=%.3f GB (phase=%d)\n",
         (double)g_peak.load() / 1e9, g_peak_phase.load());
  printf("[bench] SUMMARY build=%.1fms search=%.1fms post_reload_search=%.1fms "
         "serialize=%.1fms deserialize=%.1fms\n",
         t_build, t_search, t_search2, t_ser, t_deser);
  seekdb_cuvs_free(h);
  return 0;
}
