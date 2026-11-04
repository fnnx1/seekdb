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
| `bench_host.cpp` | end-to-end harness: build / search / serialize / reload |

```bash
./build_host.sh /path/to/libseekdb_cuvs_bridge.so
roc-obj-ls libseekdb_cuvs_bridge.so   # "No kernel section found"
```

## Design

1. `cuvsBruteForceBuild` over the fp16 dataset (device-resident, non-owning
   view) with `L2Expanded`.
2. For each chunk of 256 base rows as queries: `cuvsBruteForceSearch` top-128
   into an `int64` device buffer, async D2H on a non-blocking staging stream,
   4-deep pipeline.  cuVS runs on a dedicated non-blocking stream set with
   `cuvsStreamSet`; this is what keeps the search stream saturated (without it
   the build is ~1.7x slower because each D2H drains the queue).
3. Assemble the `n x 128` uint32 exact graph on the host and import it with
   `cuvsCagraIndexFromArgs` (+ fp32 dataset).
4. Query: CAGRA search top-256, then exact AVX2 fp32 host rerank of the 256
   candidates to the requested `topk`.
5. Snapshot: `cuvsCagraSerialize(..., include_dataset=true)` -> 384 MB;
   `deserialize` recovers the fp32 base from the snapshot for rerank and
   validates `n`/`dim`.

Relevant env knobs (defaults in parentheses): `TILED_BF_CHUNK` (256),
`TILED_BF_K` (128), `TILED_BF_HALF` (1), `TILED_BF_DEPTH` (4),
`TILED_INDEX_K` (128), `TILED_RERANK_L` (256 for dim != 128),
`TILED_TIME`, `TILED_MEMDBG`, `TILED_SELFCHECK`, `TILED_BUILD_MODE=cagra`
(experimental direct CAGRA build), `TILED_SKIP_INDEX`, `TILED_SKIP_SERIALIZE`.

## Measured on the proxy set (250K x 256, 100 queries, gfx1100)

Same session, same data, old device bridge (`build.sh`) vs host-only bridge:

| metric | old (device) | host-only | delta |
| --- | --- | --- | --- |
| .so compile | 4.47 s | 0.98 s | **-3.49 s** |
| .so size / code object | 117 KB / gfx1100 56 KB | 48 KB / none | - |
| index build | 3.32 s | 5.52 s (5.43-5.60) | +2.20 s |
| build-index search | 120 ms | 32 ms | - |
| R@10 / R@100 (build) | 1.0000 / 0.8799 | 1.0000 / 0.8799 | identical |
| R@10 / R@100 (reload) | 1.0000 / 0.8799 | 1.0000 / 0.8799 | identical |
| VRAM peak, build phase | 0.671 GB | 0.694 GB | +0.023 GB |
| VRAM peak, full run | 0.908 GB | 0.919-1.019 GB | ~same |
| serialize / deserialize | 770 / 504 ms | 760-800 / 336-373 ms | same / faster |
| snapshot size | 384 MB | 384 MB | same |

Estimate if the platform build metric covers `compile + index build` (no other
fixed cost): 7.79 s -> 6.50 s (-17%), i.e. the 2.706x build score would become
~3.2x.  If the metric covers the index build only, the host-only build is 1.66x
slower and that score regresses.

Note on recall: on this proxy data the old device bridge also scores 0.8799,
so the host-only path preserves (does not degrade) recall; the quoted 0.9015
comes from the eval data set, not from this proxy set.

## Negative / exploratory results

* Brute-force C API throughput is ~4.6 TFLOP/s effective (fp16, top-128) vs
  ~11.4 TFLOP/s for the custom tiled fp32 kernels; the exact graph therefore
  costs ~2.2 s more than the device path.
* The BF C API rejects uint32 neighbors ("neighbors should be of type
  int64_t") despite the header doc; `int64` output doubles graph D2H bytes.
* `TILED_BUILD_MODE=cagra` (direct CAGRA NN-descent build, no exact graph):
  9.27 s build and R@100 = 0.8596 - worse on both axes, so the exact graph is
  still worth its cost.
* A per-chunk synchronous D2H costs ~2.3 ms/chunk of queue drain (build 9.0 s
  instead of 5.5 s); the dual-stream pipeline is required.

## Risks

* The index-build regression is real: on a build metric that excludes compile
  time this change loses.
* Absolute timings move ~5% run to run (shared GPU).
* CAGRA-search-phase peak VRAM is above 0.8 GB for the old bridge as well; if
  the limit is measured over the whole run (not the build phase), both fail.
* `cuvsStreamSet`/non-blocking stream pipelining assumes cuVS honours the
  resource stream (validated by `TILED_SELFCHECK`, 250000/250000 self-hits).
