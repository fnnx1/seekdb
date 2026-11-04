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
| `bench_host.cpp` | end-to-end harness: build / search / serialize / reload, per-phase VRAM |

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
5. Snapshot (256 MB): custom `SKH1` header + uint32 graph + fp16 dataset.
   hipVS 26.03 `cuvsCagraDeserialize` rejects fp16 CAGRA files, so the index is
   rebuilt with `cuvsCagraIndexFromArgs` on load instead.  With
   `TILED_FP16_INDEX=0` the bridge falls back to the native fp32
   `cuvsCagraSerialize`/`Deserialize` (384 MB snapshot).

Env knobs (defaults): `TILED_BF_CHUNK` (256), `TILED_BF_K` (128),
`TILED_BF_HALF` (1), `TILED_BF_DEPTH` (4), `TILED_INDEX_K` (128),
`TILED_RERANK_L` (256), `TILED_RERANK_THREADS` (8), `TILED_FP16_INDEX` (1),
`TILED_QUERY_CHUNK` (0 = off), `TILED_TIME`, `TILED_MEMDBG`,
`TILED_SELFCHECK`, `TILED_SKIP_INDEX`, `TILED_SKIP_SERIALIZE`,
`TILED_RMM_RESET_ON_FREE` (0, see risks).

## Measured on the proxy set (250K x 256, 100 queries, gfx1100)

Same session, same data, old device bridge (`build.sh`) vs host-only bridge:

| metric | old (device) | host-only | delta |
| --- | --- | --- | --- |
| .so compile | 4.44-4.52 s | 0.96-0.98 s | **-3.5 s** |
| .so size / code object | 117 KB / gfx1100 56 KB | 48-59 KB / none | - |
| index build | 3.14-3.32 s | 5.33-5.51 s | +2.1 s |
| R@10 / R@100 (build) | 1.0000 / 0.8799 | 1.0000 / 0.8799 | identical |
| R@10 / R@100 (reload) | 1.0000 / 0.8799 | 1.0000 / 0.8799 | identical |
| query, warm (100 q) | 10.8-11.7 ms | 9.3-11.1 ms | same/better |
| snapshot | 384 MB | **256 MB** | -128 MB |
| serialize / deserialize | 770 / 500 ms | 433-474 / 247-255 ms | faster |
| VRAM peak, build phase | 0.671 GB | 0.694 GB | +0.023 GB |
| VRAM peak, build+query | 0.908 GB | 0.791 GB | -0.117 GB |
| VRAM peak, load+query (fresh process) | 0.908 GB | 0.761 GB | -0.147 GB |
| VRAM peak, full in-process cycle | 0.908 GB | 0.891 GB | -0.017 GB |

Per-phase host-only VRAM (hipMemGetInfo, 2 ms sampling): build 0.694 GB,
build+first query 0.791 GB, serialize 0.686 GB, deserialize 0.686 GB,
load+query fresh 0.761 GB.

The full in-process cycle builds, queries, serializes, frees, reloads and
queries again; ~186 MB of the first query's workspace stays cached after
`seekdb_cuvs_free` (cuVS/HIP-level, unaffected by `cuvsRMMMemoryResourceReset`
or `HSA_DISABLE_FRAGMENT_ALLOCATOR`), which is why that cycle peaks at
0.891 GB.  The old bridge shows the same effect (0.908 GB).  A build process
that never serves a query peaks at 0.694 GB, and a load+query process at
0.761 GB.

If the platform build metric covers `compile + index build` (no other fixed
cost): 7.6-7.8 s -> 6.3-6.5 s (-17%), i.e. the 2.706x build score would become
~3.2x.  If the metric covers the index build only, the host-only build is 1.65x
slower and that score regresses.

## Negative / exploratory results

* Brute-force C API throughput is ~4.6 TFLOP/s effective (fp16, top-128) vs
  ~11.4 TFLOP/s for the custom tiled fp32 kernels; the exact graph costs ~2.1 s
  more than the device path.
* Direct CAGRA NN-descent build (`TILED_BUILD_MODE=cagra`): 9.27 s and
  R@100 = 0.8596 - worse than the exact graph.
* CAGRA search params: `itopk_size` 512 / `search_width` 32 are needed for
  R@100 = 0.8799; 320/16 drops it to 0.8302, width 16 alone to 0.8758.
  CAGRA search transient VRAM grows ~1 MB/query, so `TILED_QUERY_CHUNK` can cap
  it, but every extra CAGRA call costs ~1.8 ms (100 queries chunked to 16 took
  20.6 ms vs 10.9 ms).
* `TILED_RERANK_L=160/192` still gives R@100 0.8799 on this proxy set; 256 is
  kept for margin on the eval data.

## Risks

* The index-build regression is real: on a build metric that excludes compile
  time this change loses.
* Full in-process build->reload->query peak is 0.891 GB (allocator retention,
  old bridge 0.908 GB); separate build and load processes peak at 0.694/0.761 GB.
* Absolute timings move ~5% run to run (shared GPU).
* Custom `SKH1` snapshot is bridge-owned; changing the graph degree or dtype
  requires bumping its version field.
* `cuvsStreamSet`/non-blocking stream pipelining assumes cuVS honours the
  resource stream (validated by `TILED_SELFCHECK`, 250000/250000 self-hits).
