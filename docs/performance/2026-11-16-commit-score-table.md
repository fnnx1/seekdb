# 最近十次正式提测：commit × 成绩

数据来源：平台 taskrecord 接口 + 仓库 git log，截至 2026-11-16 19:55（Asia/Shanghai）。
E 为平台“性能原始指标”，Prompt 为“Agent Prompt 原始分”；invalid 表示该轮 Prompt 被判无效。

| # | 日期 | commit | git 主题 | src 相对上一条 | 正式 E | ΔE | Prompt | ΔPrompt |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-11-11 | `f19584e` | Rewrite Agent Prompt as technical joint record | 基准（对比 9720e3e：src 未变） | 989.356540 |  | 14.5 | (前一轮 invalid) |
| 2 | 2026-11-11 | `9eaae66` | Optimize exact TopK: AVX2/FMA, filter fast path, heap | +202/-36 | 1611.282154 | +621.9 | 13.5 | -1.0 |
| 3 | 2026-11-11 | `44a4652` | Remove evaluation-process experiment from material | 无 | 1484.470300 | -126.8 | invalid | — |
| 4 | 2026-11-12 | `691bd53` | Add formal E verification instance | 无 | 1574.528127 | +90.1 | 14.5 | (前一轮 invalid) |
| 5 | 2026-11-12 | `a246a0f` | Add K-size and filter-boundary transfer evidence | 无 | 1495.208476 | -79.3 | 14.5 | +0.0 |
| 6 | 2026-11-13 | `742b6cf` | OpenMP two-thread scan, JSON/CRC fast paths | +169/-29 | 1518.604488 | +23.4 | 15.5 | +1.0 |
| 7 | 2026-11-13 | `ba4f1bf` | In-memory vector prefix, small-WAL shutdown | +58/-7 | 1681.836634 | +163.2 | invalid | — |
| 8 | 2026-11-14 | `b7f536d` | Exact pruning: quantized lower bounds + protocol | +751/-140 | 2692.094842 | +1010.3 | 18.5 | (前一轮 invalid) |
| 9 | 2026-11-16 | `ae9fc6c` | 8-row SIMD block scan, grouped equality ranges | +322/-47 | 3181.770162 | +489.7 | 23.0 | +4.5 |
| 10 | 2026-11-16 | `184e413` | Process-crash commit policy (no per-commit fsync) | +188/-23 | 3323.891232 | +142.1 | 21.0 | -2.0 |

## 关键观察

1. E 从 989 → 3324（3.4×），与本地综合性能提升同向，但幅度远小于本地倍数：
   - `9eaae66`（查询 5×）E 仅 +63%（989→1611）；
   - `b7f536d`（精确剪枝 + 协议/查询重构）E +60%（1682→2692）；
   - `ae9fc6c`（块扫描 1.43×）E +18%（2692→3182）；
   - `184e413`（写入 3.9×）E 仅 +4.5%（3182→3324）。
2. 按官方公式 f(R)=R/(1+R)（饱和），E∈[0,1]，榜单数值≈E×10000；全部 R=1 的基线恰为 5000。
3. 写入 3.9× 只换来 +4.5%，与“平台写入窗口约 130ms、其中引擎仅 ~32ms，去 fsync 只让窗口降 7%”一致：
   平台写相耗时含全部批次请求、结果输出与持久化，协议/驱动/平台存储占大头，引擎侧改动被稀释。
4. 四条 Prompt 有效分为 13.5 / 14.5 / 18.5 / 23.0 / 21.0，波动与材料版本和三次评审中位数有关；
   同一源码仅换材料时（44a4652、691bd53、a246a0f）Prompt 在 13.5–14.5 之间波动，说明存在评审方差。
5. 平台按“两项排名分之和”择优保留：`184e413` E 更高但 Prompt 21.0，两项和未超过 `ae9fc6c`（23.0），
   所以榜单仍是 `ae9fc6c`（E 3181.77 / Prompt 23.0 → 25 分，综合第 61）。
