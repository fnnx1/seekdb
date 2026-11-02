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

typedef struct {
  cuvsResources_t res;
  cuvsCagraIndex_t idx;
  long n, dim, k;
  void* d_dataset;
  void* d_graph;
} sk_bf_index;

static int env_int(const char* key, int dflt)
{
  const char* v = getenv(key);
  return v && *v ? atoi(v) : dflt;
}

static void set_tensor(DLManagedTensor* t, void* data, DLDeviceType dev, int bits,
                       long r, long c, long* shape)
{
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

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_bf_index* h = (sk_bf_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  h->k = (dim == 128) ? env_int("BF_K", 64) : env_int("BF_K", 60);

  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  if (cudaMalloc(&h->d_dataset, (size_t)n * dim * 4) != cudaSuccess) goto fail;
  if (cudaMemcpy(h->d_dataset, base, (size_t)n * dim * 4, cudaMemcpyHostToDevice) != cudaSuccess) goto fail;

  cuvsBruteForceIndex_t bf = NULL;
  if (cuvsBruteForceIndexCreate(&bf) != CUVS_SUCCESS) goto fail;
  DLManagedTensor dset; long ds[2];
  set_tensor(&dset, h->d_dataset, kDLCUDA, 32, n, dim, ds);
  if (cuvsBruteForceBuild(h->res, &dset, L2Expanded, 0.f, bf) != CUVS_SUCCESS) {
    fprintf(stderr, "[bf_cagra] bf build failed: %s\n", cuvsGetLastErrorText());
    cuvsBruteForceIndexDestroy(bf); goto fail;
  }

  long chunk = env_int("BF_CHUNK", 32768);
  if (chunk <= 0) chunk = 32768;
  if (chunk > n) chunk = n;
  int64_t* d_ids = NULL; float* d_dist = NULL;
  if (cudaMalloc((void**)&d_ids, (size_t)chunk * h->k * 8) != cudaSuccess) { cuvsBruteForceIndexDestroy(bf); goto fail; }
  if (cudaMalloc((void**)&d_dist, (size_t)chunk * h->k * 4) != cudaSuccess) { cuvsBruteForceIndexDestroy(bf); cudaFree(d_ids); goto fail; }
  int64_t* h_ids = (int64_t*)malloc((size_t)chunk * h->k * 8);
  uint32_t* h_graph = (uint32_t*)malloc((size_t)n * h->k * 4);
  if (!h_ids || !h_graph) { free(h_ids); free(h_graph); cuvsBruteForceIndexDestroy(bf); cudaFree(d_ids); cudaFree(d_dist); goto fail; }
  cuvsFilter filter; filter.type = NO_FILTER; filter.addr = (uintptr_t)0;
  int ok = 1;
  for (long start = 0; start < n; start += chunk) {
    long m = n - start; if (m > chunk) m = chunk;
    DLManagedTensor qt = dset; long qs[2] = {m, dim};
    qt.dl_tensor.data = (char*)h->d_dataset + (size_t)start * dim * 4;
    qt.dl_tensor.shape = qs;
    DLManagedTensor nb, db; long os[2] = {m, h->k};
    set_tensor(&nb, d_ids, kDLCUDA, 64, m, h->k, os);
    nb.dl_tensor.dtype.code = kDLInt;
    set_tensor(&db, d_dist, kDLCUDA, 32, m, h->k, os);
    if (cuvsBruteForceSearch(h->res, bf, &qt, &nb, &db, filter) != CUVS_SUCCESS) {
      fprintf(stderr, "[bf_cagra] bf chunk failed at %ld: %s\n", start, cuvsGetLastErrorText());
      ok = 0; break;
    }
    if (cudaMemcpy(h_ids, d_ids, (size_t)m * h->k * 8, cudaMemcpyDeviceToHost) != cudaSuccess) { ok = 0; break; }
    size_t off = (size_t)start * h->k;
    for (long i = 0; i < m * h->k; ++i) h_graph[off + i] = (uint32_t)h_ids[i];
  }
  cuvsBruteForceIndexDestroy(bf);
  cudaFree(d_ids); cudaFree(d_dist); free(h_ids);
  if (!ok) { free(h_graph); goto fail; }

  cudaFree(h->d_dataset); h->d_dataset = NULL;

  /* Keep graph on host: cuvsCagraIndexFromArgs copies host inputs to device. */

  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  DLManagedTensor graph; long gs[2];
  set_tensor(&graph, h_graph, kDLCPU, 32, n, h->k, gs);
  graph.dl_tensor.dtype.code = kDLUInt;
  DLManagedTensor hset; long hs[2];
  set_tensor(&hset, (void*)base, kDLCPU, 32, n, dim, hs);
  if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph, &hset, h->idx) != CUVS_SUCCESS) {
    fprintf(stderr, "[bf_cagra] cagra from args failed: %s\n", cuvsGetLastErrorText());
    free(h_graph); goto fail;
  }
  free(h_graph);
  fprintf(stderr, "[bf_cagra] built n=%ld dim=%ld k=%ld\n", n, dim, h->k);
  return h;

fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->d_graph) cudaFree(h->d_graph);
  if (h->d_dataset) cudaFree(h->d_dataset);
  if (h->res) cuvsResourcesDestroy(h->res);
  free(h);
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_bf_index* h = (sk_bf_index*)handle;
  if (!h) return 1;
  int rc = 0;
  float* dq = NULL; unsigned* dids = NULL; float* ddist = NULL;
  cuvsCagraSearchParams_t sp = NULL;
  if (cudaMalloc((void**)&dq, (size_t)nq * h->dim * 4) != cudaSuccess) { rc = 10; goto done; }
  if (cudaMalloc((void**)&dids, (size_t)nq * topk * 4) != cudaSuccess) { rc = 11; goto done; }
  if (cudaMalloc((void**)&ddist, (size_t)nq * topk * 4) != cudaSuccess) { rc = 12; goto done; }
  if (cudaMemcpy(dq, query, (size_t)nq * h->dim * 4, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2] = {nq, h->dim}, os[2] = {nq, topk};
    set_tensor(&qt, dq, kDLCUDA, 32, nq, h->dim, qs);
    set_tensor(&nt, dids, kDLCUDA, 32, nq, topk, os); nt.dl_tensor.dtype.code = kDLUInt;
    set_tensor(&dt, ddist, kDLCUDA, 32, nq, topk, os);
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 14; goto done; }
    if (sp->itopk_size < (size_t)env_int("BF_ITOPK", 512)) sp->itopk_size = env_int("BF_ITOPK", 512);
    sp->search_width = env_int("BF_WIDTH", 32);
    cuvsFilter filter; filter.type = NO_FILTER; filter.addr = (uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, filter) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (cudaMemcpy(out_ids, dids, (size_t)nq * topk * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist && cudaMemcpy(out_dist, ddist, (size_t)nq * topk * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 17; goto done; }
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (ddist) cudaFree(ddist);
  if (dids) cudaFree(dids);
  if (dq) cudaFree(dq);
  if (rc) fprintf(stderr, "[bf_cagra] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_bf_index* h = (sk_bf_index*)handle;
  if (!h) return;
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->d_graph) cudaFree(h->d_graph);
  if (h->d_dataset) cudaFree(h->d_dataset);
  if (h->res) cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_bf_index* h = (sk_bf_index*)handle;
  if (!h || !filename) return 1;
  if (cuvsCagraSerialize(h->res, filename, h->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[bf_cagra] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_bf_index* h = (sk_bf_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  h->n = n; h->dim = dim;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS) goto fail;
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
