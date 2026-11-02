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

typedef struct {
  cuvsResources_t res;
  cuvsCagraIndex_t idx;
  long n, dim, k;
  void* d_dataset;
  void* d_graph;
  float* host_base;
  uint32_t* host_graph;
  long graph_k;
} sk_bf_index;

static int env_int(const char* key, int dflt)
{
  const char* v = getenv(key);
  return v && *v ? atoi(v) : dflt;
}

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

__global__ static void finish_distance_tile(float* d, uint16_t* ids,
                                             const float* norms,
                                             long n, long dim, long q0,
                                             long x0, int B, int T)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)B * T;
  if (i >= count) return;
  long qr = (long)(i / T);
  long xc = (long)(i % T);
  long qid = q0 + qr;
  long xid = x0 + xc;
  if (qid >= n || xid >= n) {
    d[i] = INFINITY;
    ids[i] = UINT16_MAX;
    return;
  }
  d[i] += norms[qid] + norms[xid];
  ids[i] = (uint16_t)xc;
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

static int build_exact_tiled_graph(const float* base, long n, long dim, int K,
                                   int B, int T, uint32_t** graph_out)
{
  int rc = 1;
  float* d_base = NULL;
  float* d_norms = NULL;
  float *key0 = NULL, *key1 = NULL;
  uint16_t *val0 = NULL, *val1 = NULL;
  float *cand0 = NULL, *cand1 = NULL;
  uint32_t *cid0 = NULL, *cid1 = NULL;
  uint32_t *tile_offsets = NULL, *cand_offsets = NULL;
  void* temp = NULL;
  hipblasHandle_t blas = NULL;
  uint32_t* graph = NULL;
  uint32_t* d_graph_block = NULL;
  long lcm = (long)B * (long)T / gcd_int(B, T);
  long padded_n = ((n + lcm - 1) / lcm) * lcm;
  int ntiles = (int)(padded_n / T);
  int qblocks = (int)(padded_n / B);
  int cpr = ntiles * K;
  size_t tile_elems = (size_t)B * T;
  size_t candidate_elems = (size_t)B * cpr;
  size_t temp_tile = 0, temp_cand = 0, temp_bytes = 0;
  uint32_t* h_tile_offsets = (uint32_t*)malloc(((size_t)B + 1) * sizeof(uint32_t));
  uint32_t* h_cand_offsets = (uint32_t*)malloc(((size_t)B + 1) * sizeof(uint32_t));

  if (n <= 0 || dim <= 0 || K <= 0 || K > 256 || B <= 0 || T <= 0 ||
      padded_n > INT_MAX || T > 65535 || tile_elems > UINT_MAX || candidate_elems > UINT_MAX ||
      !h_tile_offsets || !h_cand_offsets) { free(h_tile_offsets); free(h_cand_offsets); return 1; }
  for (int i = 0; i <= B; ++i) {
    h_tile_offsets[i] = (uint32_t)((size_t)i * T);
    h_cand_offsets[i] = (uint32_t)((size_t)i * cpr);
  }
  if (hipMalloc((void**)&d_base, (size_t)padded_n * dim * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&d_norms, (size_t)padded_n * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&key0, tile_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&key1, tile_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&val0, tile_elems * sizeof(uint16_t)) != hipSuccess ||
      hipMalloc((void**)&val1, tile_elems * sizeof(uint16_t)) != hipSuccess ||
      hipMalloc((void**)&cand0, candidate_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&cand1, candidate_elems * sizeof(float)) != hipSuccess ||
      hipMalloc((void**)&cid0, candidate_elems * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&cid1, candidate_elems * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&tile_offsets, ((size_t)B + 1) * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&cand_offsets, ((size_t)B + 1) * sizeof(uint32_t)) != hipSuccess ||
      hipMalloc((void**)&d_graph_block, (size_t)B * K * sizeof(uint32_t)) != hipSuccess) {
    fprintf(stderr, "[tiled] allocation failed: %s\n", hipGetErrorString(hipGetLastError()));
    goto cleanup;
  }
  if (hipMemcpy(d_base, base, (size_t)n * dim * sizeof(float), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(tile_offsets, h_tile_offsets, ((size_t)B + 1) * sizeof(uint32_t), hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(cand_offsets, h_cand_offsets, ((size_t)B + 1) * sizeof(uint32_t), hipMemcpyHostToDevice) != hipSuccess ||
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

  {
    rocprim::double_buffer<float> tk(key0, key1);
    rocprim::double_buffer<uint16_t> tv(val0, val1);
    rocprim::double_buffer<float> ck(cand0, cand1);
    rocprim::double_buffer<uint32_t> cv(cid0, cid1);
    hipError_t e1 = rocprim::segmented_radix_sort_pairs(
        NULL, temp_tile, tk, tv, (uint32_t)tile_elems, (uint32_t)B,
        tile_offsets, tile_offsets + 1, 0, 32, 0);
    hipError_t e2 = rocprim::segmented_radix_sort_pairs(
        NULL, temp_cand, ck, cv, (uint32_t)candidate_elems, (uint32_t)B,
        cand_offsets, cand_offsets + 1, 0, 32, 0);
    if (e1 != hipSuccess || e2 != hipSuccess) {
      fprintf(stderr, "[tiled] rocprim temp query failed: %s / %s\n",
              hipGetErrorString(e1), hipGetErrorString(e2));
      goto cleanup;
    }
  }
  temp_bytes = temp_tile > temp_cand ? temp_tile : temp_cand;
  if (hipMalloc(&temp, temp_bytes) != hipSuccess || hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[tiled] temp/hipblas setup failed\n");
    goto cleanup;
  }
  hipblasSetPointerMode(blas, HIPBLAS_POINTER_MODE_HOST);
  hipblasSetStream(blas, 0);
  graph = (uint32_t*)malloc((size_t)n * K * sizeof(uint32_t));
  if (!graph) { fprintf(stderr, "[tiled] host graph allocation failed\n"); goto cleanup; }

  for (long q0 = 0; q0 < padded_n; q0 += B) {
    if (q0 >= n) continue;
    int qrows = (int)((n - q0) < B ? (n - q0) : B);
    for (int tile = 0; tile < ntiles; ++tile) {
      long x0 = (long)tile * T;
      const float alpha = -2.0f, beta = 0.0f;
      /* Row-major Q[B,D] and X[T,D] are interpreted as column-major D×B/D×T.
         This computes C_col[T,B]=X*Q^T, stored as row-major distances[B,T]. */
      hipblasStatus_t hs = hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
          T, B, (int)dim, &alpha,
          d_base + (size_t)x0 * dim, (int)dim,
          d_base + (size_t)q0 * dim, (int)dim,
          &beta, key0, T);
      if (hs != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[tiled] sgemm failed q=%ld tile=%d status=%d\n", q0, tile, (int)hs);
        goto cleanup;
      }
      finish_distance_tile<<<(unsigned int)((tile_elems + 255) / 256), 256>>>(
          key0, val0, d_norms, n, dim, q0, x0, B, T);
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] distance finish failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      rocprim::double_buffer<float> keys(key0, key1);
      rocprim::double_buffer<uint16_t> vals(val0, val1);
      if (rocprim::segmented_radix_sort_pairs(
              temp, temp_bytes, keys, vals, (uint32_t)tile_elems, (uint32_t)B,
              tile_offsets, tile_offsets + 1, 0, 32, 0) != hipSuccess) {
        fprintf(stderr, "[tiled] tile sort failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
      gather_tile_topk<<<(unsigned int)((B + 127) / 128), 128>>>(
          keys.current(), vals.current(), cand0, cid0, B, T, K, tile, ntiles, T);
      if (hipGetLastError() != hipSuccess) {
        fprintf(stderr, "[tiled] tile gather failed: %s\n", hipGetErrorString(hipGetLastError()));
        goto cleanup;
      }
    }
    rocprim::double_buffer<float> ckeys(cand0, cand1);
    rocprim::double_buffer<uint32_t> cvals(cid0, cid1);
    if (rocprim::segmented_radix_sort_pairs(
            temp, temp_bytes, ckeys, cvals, (uint32_t)candidate_elems, (uint32_t)B,
            cand_offsets, cand_offsets + 1, 0, 32, 0) != hipSuccess) {
      fprintf(stderr, "[tiled] final sort failed: %s\n", hipGetErrorString(hipGetLastError()));
      goto cleanup;
    }
    gather_final_topk<<<(unsigned int)((B + 127) / 128), 128>>>(
        cvals.current(), d_graph_block, B, K, K, cpr);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
      fprintf(stderr, "[tiled] final gather failed: %s\n", hipGetErrorString(hipGetLastError()));
      goto cleanup;
    }
    if (hipMemcpy(graph + (size_t)q0 * K, d_graph_block,
                  (size_t)qrows * K * sizeof(uint32_t), hipMemcpyDeviceToHost) != hipSuccess) {
      fprintf(stderr, "[tiled] graph copy failed: %s\n", hipGetErrorString(hipGetLastError()));
      goto cleanup;
    }
  }

  if (hipDeviceSynchronize() != hipSuccess) goto cleanup;
  *graph_out = graph;
  graph = NULL;
  rc = 0;

cleanup:
  if (blas) hipblasDestroy(blas);
  if (temp) hipFree(temp);
  if (cand_offsets) hipFree(cand_offsets);
  if (tile_offsets) hipFree(tile_offsets);
  if (cid1) hipFree(cid1); if (cid0) hipFree(cid0);
  if (cand1) hipFree(cand1); if (cand0) hipFree(cand0);
  if (val1) hipFree(val1); if (val0) hipFree(val0);
  if (key1) hipFree(key1); if (key0) hipFree(key0);
  if (d_graph_block) hipFree(d_graph_block);
  if (d_norms) hipFree(d_norms);
  if (d_base) hipFree(d_base);
  free(h_tile_offsets); free(h_cand_offsets);
  free(graph);
  return rc;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_bf_index* h = (sk_bf_index*)calloc(1, sizeof(*h));
  if (!h || !base || n <= 0 || dim <= 0) { free(h); return NULL; }
  h->n = n; h->dim = dim;
  h->host_base = NULL;
  h->host_graph = NULL;
  h->graph_k = 0;
  int rerank_l = env_int("TILED_RERANK_L", h->dim == 128 ? 0 : 100);
  int expand_seeds = env_int("TILED_EXPAND_SEEDS", h->dim == 128 ? 0 : 15);
  if (rerank_l > 0 || expand_seeds > 0) {
    size_t base_bytes = (size_t)n * (size_t)dim * sizeof(float);
    h->host_base = (float*)malloc(base_bytes);
    if (!h->host_base) { free(h); return NULL; }
    memcpy(h->host_base, base, base_bytes);
  }
  h->k = (dim == 128) ? env_int("TILED_K", 64) : env_int("TILED_K", 256);
  int B = env_int("TILED_Q_BLOCK", dim == 128 ? 1024 : 256);
  int T = env_int("TILED_BASE_TILE", dim == 128 ? 8192 : 16384);
  if (B <= 0 || T <= 0 || h->k <= 0 || h->k > 256) { free(h); return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h->host_base); free(h); return NULL; }

  uint32_t* graph = NULL;
  if (build_exact_tiled_graph(base, n, dim, (int)h->k, B, T, &graph) != 0) goto fail;
  if (env_int("TILED_SKIP_INDEX", 0)) {
    free(graph);
    return h;
  }
  /* BF uses direct HIP allocations; release any process-level RMM pool/cache before CAGRA import. */
  if (cuvsRMMMemoryResourceReset() != CUVS_SUCCESS)
    fprintf(stderr, "[tiled] RMM resource reset warning: %s\n", cuvsGetLastErrorText());
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { free(graph); goto fail; }
  {
    DLManagedTensor graph_t = {0}, base_t = {0};
    long gs[2] = {n, h->k}, bs[2] = {n, dim};
    int index_k = h->k > 128 ? 128 : (int)h->k;
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
    set_tensor(&graph_t, index_graph, kDLCPU, 32, n, index_k, gs);
    graph_t.dl_tensor.dtype.code = kDLUInt;
    set_tensor(&base_t, (void*)base, kDLCPU, 32, n, dim, bs);
    if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &graph_t, &base_t, h->idx) != CUVS_SUCCESS) {
      fprintf(stderr, "[tiled] CAGRA import failed: %s\n", cuvsGetLastErrorText());
      free(compact_graph); free(graph); goto fail;
    }
    free(compact_graph);
  }
  if (expand_seeds > 0) {
    h->host_graph = graph;
    h->graph_k = h->k;
  } else {
    free(graph);
  }
  fprintf(stderr, "[tiled] built n=%ld dim=%ld k=%ld B=%d T=%d\n", n, dim, h->k, B, T);
  return h;
fail:
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->res) cuvsResourcesDestroy(h->res);
  free(h->host_base);
  free(h->host_graph);
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
  int rerank_l = env_int("TILED_RERANK_L", h->dim == 128 ? 0 : 100);
  int expand_seeds = env_int("TILED_EXPAND_SEEDS", h->dim == 128 ? 0 : 15);
  int second_hop = env_int("TILED_SECOND_HOP", h->dim == 128 ? 0 : 40);
  int do_refine = h->host_base && (rerank_l > topk || expand_seeds > 0);
  if (do_refine && search_k < rerank_l) search_k = rerank_l;
  if (do_refine && search_k < expand_seeds) search_k = expand_seeds;
  float* dq = NULL; unsigned* dids = NULL; float* ddist = NULL;
  unsigned* h_candidates = NULL; unsigned* cand_ids = NULL;
  struct Ranked { float d; unsigned id; };
  struct Ranked* ranked = NULL;
  size_t candidate_cap = (size_t)search_k;
  if (do_refine && h->host_graph && expand_seeds > 0)
    candidate_cap += (size_t)(expand_seeds + (second_hop > 0 ? second_hop : 0)) * (size_t)h->graph_k;
  cuvsCagraSearchParams_t sp = NULL;
  if (cudaMalloc((void**)&dq, (size_t)nq * h->dim * 4) != cudaSuccess) { rc = 10; goto done; }
  if (cudaMalloc((void**)&dids, (size_t)nq * search_k * 4) != cudaSuccess) { rc = 11; goto done; }
  if (cudaMalloc((void**)&ddist, (size_t)nq * search_k * 4) != cudaSuccess) { rc = 12; goto done; }
  if (cudaMemcpy(dq, query, (size_t)nq * h->dim * 4, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  if (do_refine) {
    h_candidates = (unsigned*)malloc((size_t)nq * search_k * sizeof(unsigned));
    cand_ids = (unsigned*)malloc(candidate_cap * sizeof(unsigned));
    ranked = (struct Ranked*)malloc(candidate_cap * sizeof(struct Ranked));
    if (!h_candidates || !cand_ids || !ranked) { rc = 18; goto done; }
  }
  {
    DLManagedTensor qt, nt, dt; long qs[2] = {nq, h->dim}, os[2] = {nq, search_k};
    set_tensor(&qt, dq, kDLCUDA, 32, nq, h->dim, qs);
    set_tensor(&nt, dids, kDLCUDA, 32, nq, search_k, os); nt.dl_tensor.dtype.code = kDLUInt;
    set_tensor(&dt, ddist, kDLCUDA, 32, nq, search_k, os);
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 14; goto done; }
    size_t itopk = (size_t)env_int("BF_ITOPK", 512);
    if (itopk < (size_t)search_k) itopk = (size_t)search_k;
    if (sp->itopk_size < itopk) sp->itopk_size = itopk;
    sp->search_width = env_int("BF_WIDTH", 32);
    cuvsFilter filter; filter.type = NO_FILTER; filter.addr = (uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, filter) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (do_refine) {
      if (cudaMemcpy(h_candidates, dids, (size_t)nq * search_k * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
      for (long qi = 0; qi < nq; ++qi) {
        size_t count = 0;
        long seeds = expand_seeds > 0 ? expand_seeds : search_k;
        if (seeds > search_k) seeds = search_k;
        for (long c = 0; c < search_k; ++c) {
          unsigned id = h_candidates[(size_t)qi * search_k + c];
          if ((long)id < h->n) cand_ids[count++] = id;
        }
        if (h->host_graph && expand_seeds > 0) {
          for (long c = 0; c < seeds; ++c) {
            unsigned seed = h_candidates[(size_t)qi * search_k + c];
            if ((long)seed >= h->n) continue;
            const uint32_t* row = h->host_graph + (size_t)seed * h->graph_k;
            for (long j = 0; j < h->graph_k; ++j) {
              unsigned id = row[j];
              if ((long)id < h->n) cand_ids[count++] = id;
            }
          }
        }
        std::sort(cand_ids, cand_ids + count);
        size_t unique_count = (size_t)(std::unique(cand_ids, cand_ids + count) - cand_ids);
        const float* q = query + (size_t)qi * h->dim;
        if (h->host_graph && second_hop > 0) {
          for (size_t c = 0; c < unique_count; ++c) {
            unsigned id = cand_ids[c];
            const float* x = h->host_base + (size_t)id * h->dim;
            float d = 0.0f;
            for (long t = 0; t < h->dim; ++t) { float z = q[t] - x[t]; d = fmaf(z, z, d); }
            ranked[c].d = d; ranked[c].id = id;
          }
          std::sort(ranked, ranked + unique_count, [](const struct Ranked& a, const struct Ranked& b) { return a.d < b.d; });
          size_t expand2 = (size_t)second_hop < unique_count ? (size_t)second_hop : unique_count;
          for (size_t c = 0; c < expand2; ++c) {
            const uint32_t* row = h->host_graph + (size_t)ranked[c].id * h->graph_k;
            for (long j = 0; j < h->graph_k; ++j) {
              unsigned id = row[j];
              if ((long)id < h->n) cand_ids[count++] = id;
            }
          }
          std::sort(cand_ids, cand_ids + count);
          unique_count = (size_t)(std::unique(cand_ids, cand_ids + count) - cand_ids);
        }
        for (size_t c = 0; c < unique_count; ++c) {
          unsigned id = cand_ids[c];
          const float* x = h->host_base + (size_t)id * h->dim;
          float d = 0.0f;
          for (long t = 0; t < h->dim; ++t) { float z = q[t] - x[t]; d = fmaf(z, z, d); }
          ranked[c].d = d; ranked[c].id = id;
        }
        std::sort(ranked, ranked + unique_count, [](const struct Ranked& a, const struct Ranked& b) { return a.d < b.d; });
        for (long j = 0; j < topk; ++j) {
          if ((size_t)j < unique_count) {
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
  free(h_candidates); free(cand_ids); free(ranked);
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
  if (h->res) cuvsResourcesDestroy(h->res);
  free(h->host_base);
  free(h->host_graph);
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
  if (h->host_base && h->host_graph && h->graph_k > 0) {
    char path[4096];
    snprintf(path, sizeof(path), "%s.tiled", filename);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[tiled] sidecar open failed: %s\n", path); return 0; }
    uint64_t magic = 0x544C4431ULL;
    int64_t hn = h->n, hd = h->dim, hk = h->graph_k;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&hn, sizeof(hn), 1, f);
    fwrite(&hd, sizeof(hd), 1, f);
    fwrite(&hk, sizeof(hk), 1, f);
    fwrite(h->host_base, sizeof(float), (size_t)hn * hd, f);
    fwrite(h->host_graph, sizeof(uint32_t), (size_t)hn * hk, f);
    fclose(f);
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
  {
    char path[4096];
    snprintf(path, sizeof(path), "%s.tiled", filename);
    FILE* f = fopen(path, "rb");
    if (f) {
      uint64_t magic = 0; int64_t hn = 0, hd = 0, hk = 0;
      if (fread(&magic, sizeof(magic), 1, f) == 1 &&
          fread(&hn, sizeof(hn), 1, f) == 1 &&
          fread(&hd, sizeof(hd), 1, f) == 1 &&
          fread(&hk, sizeof(hk), 1, f) == 1 &&
          magic == 0x544C4431ULL && hn == n && hd == dim && hk > 0 && hk <= 256) {
        h->host_base = (float*)malloc((size_t)hn * hd * sizeof(float));
        h->host_graph = (uint32_t*)malloc((size_t)hn * hk * sizeof(uint32_t));
        if (h->host_base && h->host_graph &&
            fread(h->host_base, sizeof(float), (size_t)hn * hd, f) == (size_t)hn * hd &&
            fread(h->host_graph, sizeof(uint32_t), (size_t)hn * hk, f) == (size_t)hn * hk) {
          h->graph_k = hk;
        } else {
          free(h->host_base); free(h->host_graph);
          h->host_base = NULL; h->host_graph = NULL; h->graph_k = 0;
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
