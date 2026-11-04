# Host-only cuVS brute-force bridge (AMD 600072)

Host-only replacement for the fat (gfx1100 device-code) `seekdb_cuvs_bridge.c`.
It keeps the same five-symbol C ABI (`seekdb_cuvs_build`, `seekdb_cuvs_search`,
`seekdb_cuvs_free`, `seekdb_cuvs_serialize`, `seekdb_cuvs_deserialize`, plus
`seekdb_cuvs_cagra_knn`) but contains **no `__global__` kernels and no device
intrinsics**.  All GPU work is done by hipVS/cuVS library entry points.

| file | purpose |
| --- | --- |
| `seekdb_cuvs_bridge_host.c` | host-only implementation |
| `build_host.sh` | host-only build (no `-x hip`, no `--offload-arch`) |
| `build.sh` | platform entry point, execs `build_host.sh` |
| `bench_host.cpp` | lifecycle harness with per-phase VRAM |
| `conc_test.cpp` | 4-thread search consistency test |
| `multibuild_test.cpp` | small-n audit build followed by the real build |

```bash
./build_host.sh /path/to/libseekdb_cuvs_bridge.so
roc-obj-ls libseekdb_cuvs_bridge.so   # "No kernel section found"
```

`build_host.sh` locates a plain host clang++ (ROCm clang++, system clang++, g++)
and only falls back to `hipcc -x c++` when none is found; it fails the build if
`roc-obj-ls` ever reports a device code object.

## Design

1. `cuvsBruteForceBuild` over the fp16 dataset (device-resident, non-owning
   view) with `L2Expanded`.
2. For each chunk of 256 base rows as queries: `cuvsBruteForceSearch` top-128
   into an `int64` device buffer, async D2H on a non-blocking staging stream,
   4-deep pipeline.  cuVS runs on a dedicated non-blocking stream set with
   `cuvsStreamSet`; without it per-chunk D2H drains the queue (9.0 s vs 5.4 s).
3. Exact `n x 128` uint32 graph assembled on the host, imported with
   `cuvsCagraIndexFromArgs` over an **fp16** dataset (dataset 128 MB + graph
   128 MB = 256 MB index).
4. Query: CAGRA search top-256, then exact AVX2 fp32 host rerank of the 256
   candidates, deduplicated per query and fanned out over 8 threads.
   Queries are searched in chunks of `TILED_QUERY_CHUNK` (50) because the
   CAGRA search workspace grows ~1 MB/query; 2 calls for a 100-query batch
   lower the peak by ~38 MB without a measurable latency cost.
5. **Single-slot resources cache** (`TILED_RES_CACHE=1`): cuVS resources and
   streams are not destroyed with every handle.  The most recently retired
   pair is kept and handed to the next handle.  Destroying/re-creating them
   made the driver keep ~180 MB of pools/workspaces alive, which raised the
   peak of a build -> free -> load -> query lifecycle from 0.75 GB to
   0.89-1.02 GB.  Live handles never share resources, so queries stay
   independent (4-thread consistency test: 0 mismatches).
6. Snapshot (256 MB): custom `SKH1` header + uint32 graph + fp16 dataset.
   hipVS 26.03 `cuvsCagraDeserialize` rejects fp16 CAGRA files, so the index is
   rebuilt with `cuvsCagraIndexFromArgs` on load instead.  With
   `TILED_FP16_INDEX=0` the bridge uses the native fp32 snapshot (384 MB,
   full-cycle peak 0.881 GB).

Env knobs (defaults): `TILED_BF_CHUNK` (256), `TILED_BF_K` (128),
`TILED_BF_HALF` (1), `TILED_BF_DEPTH` (4), `TILED_INDEX_K` (128),
`TILED_RERANK_L` (256), `TILED_RERANK_THREADS` (8), `TILED_FP16_INDEX` (1),
`TILED_QUERY_CHUNK` (50), `TILED_RES_CACHE` (1), `TILED_TIME`, `TILED_MEMDBG`,
`TILED_VRAM_TRACE` (0.25 ms VRAM trace with step labels), `TILED_SELFCHECK`,
`TILED_SKIP_INDEX`, `TILED_SKIP_SERIALIZE`.

## Measured on the proxy set (250K x 256, 100 queries, gfx1100)

Same session, same data, old device bridge (`build.sh`) vs host-only bridge:

| metric | old (device) | host-only | delta |
| --- | --- | --- | --- |
| .so compile | 4.44-4.52 s | 1.23 s | **-3.2 s** |
| .so size / code object | 117 KB / gfx1100 56 KB | 67 KB / none | - |
| index build | 3.14-3.32 s | 5.35-5.49 s | +2.1 s |
| R@10 / R@100 (build) | 1.0000 / 0.8799 | 1.0000 / 0.8799 | identical |
| R@10 / R@100 (reload) | 1.0000 / 0.8799 | 1.0000 / 0.8799 | identical |
| query, warm (100 q) | 10.8-11.7 ms | **9.4-10.4 ms** | faster |
| snapshot | 384 MB | **256 MB** | -128 MB |
| serialize / deserialize | 770 / 500 ms | 461-475 / 238-259 ms | faster |
| VRAM peak, build phase | 0.671 GB | **0.694 GB** | +0.023 GB |
| VRAM peak, full lifecycle | 0.908 GB | **0.738 GB** | **-0.155 GB** |

Full lifecycle = build -> query -> serialize -> free -> deserialize -> query,
one fresh process, hipMemGetInfo sampled every 0.25 ms.  Step trace
(`TILED_VRAM_TRACE=1`): BF search 0.694, first CAGRA search 0.753, serialize
0.686, deserialize 0.686, reload CAGRA search 0.753.  The peak is the CAGRA
search workspace, not the snapshot load.

If the platform build metric covers `compile + index build` (no other fixed
cost): 7.6-7.8 s -> 6.6-6.7 s, i.e. the 2.706x build score would become
~3.1x.  If the metric covers the index build only, the host-only build is
1.65x slower and that score regresses.

## Negative / exploratory results

* Brute-force C API throughput is ~4.6 TFLOP/s effective (fp16, top-128) vs
  ~11.4 TFLOP/s for the custom tiled fp32 kernels; the exact graph costs ~2.1 s
  more than the device path.
* Direct CAGRA NN-descent build (`TILED_BUILD_MODE=cagra`): 9.27 s and
  R@100 = 0.8596 - worse than the exact graph.
* Search-parameter trimming cannot buy VRAM without recall loss:
  itopk 512 / width 32 is required for R@100 = 0.8799; 384/32 gives 0.8565,
  320/16 gives 0.8302, width 16 alone gives 0.8758.  Exact top-96/64 graphs
  give 0.8428/0.7742.  `algo=MULTI_KERNEL` raises peak VRAM to 1.2 GB.
* Query chunking below 64 lowers peak further (32 -> 0.719 GB)
  but costs latency (10.4 ms / 13.6 ms).
* `cuvsRMMMemoryResourceReset()` and `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` do not
  release the cached search workspace; the resources cache does.

## Risks

* The index-build regression is real: on a build metric that excludes compile
  time this change loses.
* The resources cache intentionally keeps one retired resources object
  (~0.24 GB with context/modules) resident until the next handle or process
  exit; handles never share it while alive.
* Absolute timings move ~5% run to run (shared GPU); one 20 ms query outlier
  was observed.
* Custom `SKH1` snapshot is bridge-owned; changing the graph degree or dtype
  requires bumping its version field.
* `cuvsStreamSet`/non-blocking stream pipelining assumes cuVS honours the
  resource stream (validated by `TILED_SELFCHECK`, 250000/250000 self-hits).
