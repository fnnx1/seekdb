# 2026-11-17：评分口径、字符串过滤与 WAL 页缓存优化

本轮已实现并保留两项优化：VARCHAR 等值索引/量化分组，以及不增加默认 `fdatasync` 的 WAL 页缓存回收。**尚未证明平台综合 E 提升 1.5×。** 原生 VARCHAR 负载可快数倍，但 Docker 往返会明显稀释收益；当前没有新的正式提测成绩。

## 官方依据与目标

- [600081 官方任务说明](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/lg8enxlmwg69unt6)，[本地正文](2026-11-17-official-spec.txt)。问题 API 对应性能子题 800517，见 [来源记录](2026-11-17-question-provenance.json)。
- 默认负载为 **10,000 行、128D、100 批 × 100 行、1,000 条 exact Top10，半数普通列等值过滤**。精确冻结的 schema、过滤列类型、分布/生成器和六个基线数值仍未取得。
- 查询 T 不含装载、索引准备和统一预热；B/M 包括这些阶段及最终相关写回。B 为专用数据设备 cgroup `io.stat` 的 rbytes+wbytes；M 为整个容器 `memory.peak`，**包括文件页缓存**。事后正确性校验不计入成绩。
- 故障模型限定为进程被杀、内核继续运行，不模拟断电。ACK 前完整 WAL 和提交标记仍须成功 `fflush`。
- 官方通用说明中存在一段过时的协议描述；实际冻结 JSONL 题包/驱动优先。本轮不改协议，也不把 exact 静默改成 ANN。

此前“大表可能解释平台瓶颈”的推测缺乏默认负载依据。昨日 300k/1M 的扫描收益仍成立，但不计作本轮默认 10k 的提分证据。

[2026-11-17 22:28:04 +08:00 榜单快照](2026-11-17-leaderboard.json)第 50 名为 4835.919331。用户最近记录 3323.891232，达到当时门槛需 **1.454897×**；排名会变化。

## 为什么耗时加速不能等比例提分

每个项目的 T/B/M 先独立归一化、饱和，再加权：

```text
R_X = (X_base + eps_X) / (X + eps_X)
f(R) = R / (1 + R)
eps_T = 1 ms；eps_B = eps_M = 1 MiB
E_write = .6 f(R_Tw) + .2 f(R_Bw) + .2 f(R_Mw)
E_query = .6 f(R_Tq) + .2 f(R_Bq) + .2 f(R_Mq)
E = (E_write + 2 E_query) / 3
```

有效权重：写入 T 0.2、查询 T 0.4、写入 B/M 各 1/15、查询 B/M 各 2/15。三轮分别计算 E 后取中位数，不能拼接不同轮次最好的指标。

因此，仅写入 T 快 4×，任意未知基线下的 E 绝对增量也不超过 0.066667；查询 T 快 1.5×，增量不超过 0.040408。证明：效率比乘 s 后，`max_R[f(sR)-f(R)] = (sqrt(s)-1)/(sqrt(s)+1)`，再乘该项权重。eps 存在时真实 s 应使用 `(old+eps)/(new+eps)`。

上面的界限使用官方 [0,1] 单位。榜单约为该值乘 10000 是用户记录中的显示比例推断，官方正文未明确此显示换算；不据此虚构六个基线或本轮正式 E。

已确认的解释包括：写入 T 权重仅 0.2、饱和函数压缩收益、默认免同步保留更多 WAL 页缓存，以及原生管道与 Docker 转发成本不同。**目前不能把其中任何一项量化为上次正式提测仅涨 4.47% 的唯一原因。** 正式版本差异也不只是 fsync；历史同源码分数波动不证明这次涨幅就是噪声。

## 本轮保留的实现

1. VARCHAR 与 BIGINT 同型等值过滤使用按表版本失效的 hash index，避免每次扫描全部普通列、复制字符串。没有非主键 BIGINT 时，128D 量化块按 VARCHAR 分组，让过滤后扫描使用已有八路保守下界。跨类型比较保留原逻辑。mixed schema 保留原来的 BIGINT 分组优先级。
2. 默认模式每新增约 1 MiB WAL，在成功 `fflush` 后对完整前缀调用 `POSIX_FADV_DONTNEED`。Linux 可以发起写回，但未写回脏页不会被丢弃；后续重复覆盖前缀以回收已经干净的页，保留部分尾页。advice 失败只损失缓存收益。strict 模式仍逐次同步。

源码增量见 [implementation.diff](2026-11-17-implementation.diff)。基线为 `184e413` 加昨日已验证但尚未提交的文件扫描缓冲；最终源码、native/container 二进制 SHA256 见 [manifest](2026-11-17-manifest.json)。本轮没有混入 COW、周期同步或近似检索。

## 最终版本测量

同机、同编译选项 `-O2 -std=c++17 -fopenmp`。引擎固定 CPU 0/1，驱动 CPU 2；性能实验顺序运行，完整回归在性能实验结束后运行。准备请求和答案不计入 T，预热一遍完整查询；每轮交替候选顺序，使用全新数据目录。表中每个数值为对应指标的中位数，**不是算出的正式 E**。

原生管道，5 轮，随机 1/16 倍数 float32、独立整数全扫描 oracle，杀进程重启后再次校验：

| 过滤列 | 指标 | 基线 | 最终版本 | 基线/最终 |
|---|---|---:|---:|---:|
| VARCHAR | 1000 次查询 | 109.410 ms | 23.070 ms | **4.74×** |
| VARCHAR | 写入 + 查询 | 140.373 ms | 53.211 ms | **2.64×** |
| VARCHAR | 写入 | 29.553 ms | 30.504 ms | 0.97× |
| BIGINT | 1000 次查询 | 22.835 ms | 23.400 ms | 0.98× |
| BIGINT | 写入 | 29.458 ms | 30.204 ms | 0.98× |

BIGINT 对照未获得耗时提升；约 2%–3% 的中位数差异和原始离群轮次如实保留。VARCHAR 收益依赖过滤列类型和 schema，不能假定正式负载一定命中。原始记录：[BIGINT](2026-11-17-final-bigint.json)、[VARCHAR](2026-11-17-final-varchar.json)。

Docker/cgroup，3 轮，每个写入/查询项目单独启动容器，2 CPU 配额、2 GiB 内存且无 swap。计时后宿主机同步数据文件，容器保留到资源采集结束；只选物理设备 259:0，不叠加父子设备：

| 过滤列 | 指标 | 基线 | 最终版本 |
|---|---|---:|---:|
| BIGINT | 写入 T | 58.928 ms | 60.300 ms |
| BIGINT | 查询 T | 131.569 ms | 123.301 ms |
| BIGINT | 写入 M | 14.60 MiB | **11.05 MiB** |
| BIGINT | 查询 M | 16.60 MiB | **12.52 MiB** |
| VARCHAR | 写入 T | 60.683 ms | 60.907 ms |
| VARCHAR | 查询 T | 145.657 ms | **127.688 ms（1.14×）** |
| VARCHAR | 写入 M | 14.91 MiB | **12.54 MiB** |
| VARCHAR | 查询 M | 16.37 MiB | **12.77 MiB** |
| 两种 | 各项目 B 中位数 | 5,849,088 bytes | 5,849,088 bytes |

原始记录：[BIGINT](2026-11-17-final-cgroup-bigint.json)、[VARCHAR](2026-11-17-final-cgroup-varchar.json)。每轮的资源快照、dirty/writeback、CPU quota throttling、设备统计和答案 digest 均保留。当前这些容器轮次没有 CPU quota throttling。

Docker 查询 T 波动明显，且加入了 attach 转发；它与原生 23 ms 不能混算。最终 VARCHAR 的 CPU/检索收益被往返成本稀释。若把本地容器的中位数比例视为固定比例，即使忽略写入 T 的退步，按上述饱和函数逐项求最有利基线，E 增量上界也只有约 **0.0235**。这是对这些本地比例的数学上界，**不是正式 E 预测**，更不是 1.5× 提分证明。

本地镜像与 data 共用物理设备，冷二进制页可能被计入 B；新目录并不等于官方统一冷缓存流程。驱动、编译和事后验证在被测 cgroup 外。正式专用设备、冻结生成器、基线和准备上限均未完全复刻。

普通八位小数 `uniform(-1,1)` 控制实验也测得异步回收有效：写入 M 15.18→11.58 MiB、查询 M 16.09→12.51 MiB，B 不变；写入 T 54.53→54.81 ms、查询 T 125.98→125.53 ms。这里答案由明确记录的旧二进制对拍，**不是独立浮点 oracle**。见 [原始记录](2026-11-17-cgroup-decimal.json) 和 [复现 wrapper](2026-11-17-cgroup-decimal-bench.py)。

## 已排除的候选

- 每 1 MiB `fdatasync` 再回收：M 下降，但同步增加写入 T 和物理 B；异步 advice 更合适。
- 取消 NOCOW：btrfs zstd 对 dyadic 数据物理 B 看似可少约 64%；普通随机浮点仅少 6.23%，组合原生写入还曾从约 30 ms 增到 40 ms，因此没有合入。这不是通用的 2.7× I/O 优化。
- 4-bit packed codes：少约 0.7 MiB，但查询回退至约 35 ms，排除。
- INSERT AST 简化、7-bit 非对称量化、pair8 dot、alias plan cache：未建立足够稳定或显著的新增收益，排除。pair8 安全范围与 7-bit 不兼容，未组合。
- 文件写缓冲/直接 snapshot vector view：大表改善有限，默认 10k 不命中，排除。

各轮探索原始记录以 `2026-11-17-experiment-*`、`cgroup-storage/async/varchar/decimal.json` 保留。最终表只引用最终源码的测量。

## 正确性与复现

`MVDB_SYNC_COMMIT=0 taskset -c 18,19 ./scripts/check.sh` 已全部通过，
包括原有动态/精确剪枝/440 模板/WAL/同步策略/模糊/大表/真实 spill、
新增两组回归、ASan/UBSan 和 host JSONL 20 个公开用例。
日志见 [check.txt](2026-11-17-check.txt)。最终 Docker 镜像构建及
20 个公开协议用例也全部通过，见 [Docker 日志](2026-11-17-docker-check.txt)。
最终源码与测试、测量二进制
SHA256 已重新核对一致。新增回归包括：

- 494 条独立 exact oracle 查询：Unicode、NUL、空/长字符串、转义、多个字符串列、跨类型、反向等式、ties/分页、更新删除、失败语句、事务回滚、SIGKILL 和快照恢复。
- 四组 WAL advice 回归：真实跨 1 MiB 门槛、完整页范围、重复前缀、advice 返回 EINVAL、strict 同步、ACK 后不解析 JSON 就立即杀进程、恢复后计数重置。

```bash
MVDB_SYNC_COMMIT=0 ./scripts/check.sh
MVDB_SYNC_COMMIT=0 python3 tests/performance_bench.py \
  --binary /path/to/baseline --binary ./minivectordb \
  --filter-type varchar --warmup 1 --rounds 5 --cpus 0,1 --driver-cpu 2 \
  --json /tmp/varchar.json
MVDB_SYNC_COMMIT=0 python3 tests/cgroup_performance_bench.py \
  --binary /path/to/container-baseline --binary /path/to/container-candidate \
  --filter-type varchar --warmup 1 --rounds 3 --driver-cpu 2 \
  --io-device 259:0 --json /tmp/cgroup.json
```

容器二进制须在对应镜像内编译；本机较新 glibc 二进制未必能在镜像运行。设备编号应按实际机器选择。

下一步最有信息量的是取得正式六项 T/B/M 和冻结 schema，再决定是否优化多普通列的实际过滤分组、查询投影/alias 或继续压缩持久化内存副本。当前固定选择第一非主键 BIGINT/首 VARCHAR 分组，其他过滤列会退回逐行量化路径，尚有可实验空间，但本轮未实现、未计入收益。不能承诺现有组合足够进入前 50，也没有消耗正式提测次数。
