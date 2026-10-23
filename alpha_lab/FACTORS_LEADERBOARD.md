# CZSC AlphaLab 全景因子排行榜与生态位地图 (`FACTORS_LEADERBOARD.md`)

> **自动生成时间**：`2026-10-23 00:39:05 UTC` | **台账后端**：`postgres` | **标的池**：`Full-60 Commodities (6 大板块 60 品种)`
> **生成命令**：`uv run --no-sync python -m alpha_lab.cli report`

---

## 一、 台账总览与工程基线指标

| 指标维度 | 统计数值 | 说明 |
| :--- | :---: | :--- |
| **Rust 原生 K 线信号总数 (`rust_signals`)** | `234` | 覆盖 17 个 Rust 信号子模块，含镜像对称审计标记 |
| **已消费 K 线信号数** | `50 / 234` (`21.37%`) | 50 个显式状态键 + 7 类结构强度信号 |
| **待开采精确镜像对称信号 (`exact_mirror=True`)** | `118` | 已通过仿射反转严格对称检验、尚未被现有因子消费的蓝海信号 |
| **SSOT 原子因子族 (`ATOMIC_FACTOR_DEFS`)** | `28` | 覆盖 Stage1_Base (16) + Stage2_Rust (4) + Stage3_OI (4) + Stage4_CE (4) |
| **登记历史候选因子总数 (`factor_defs`)** | `165` | 含 28 个原子族 + 137 个跨周期异构候选变体 |
| **因子↔信号血缘边数 (`factor_signal_lineage`)** | `124` | AST 静态扫描自动提取的 `state`/`strength` 依赖边 |
| **预检门禁通过率 (`preflight_audits`)** | `28 / 28` (`100%`) | `<1s` 仿射镜像对称 + 前视零泄漏 + 边界合法性门禁 |
| **品种级评估明细行 (`factor_asset_scores`)** | `55,680` | Full-60 品种 × 多周期 × Fold1_2025 / Fold2_2026 样本外打分 |
| **全池准入汇总行 (`factor_pool_scores`)** | `414` | Gate 1B 全品种池准入检验汇总 |
| **组合消融归因记录 (`ablation_runs`)** | `18` | Stage 1 ~ Stage 4 等权 / 机构两层风险预算组合样本外净值 |
| **失败实验墓地记录 (`factor_graveyard`)** | `95` | 被 Gate 1B/2/3 或 Pre-flight 淘汰的证伪候选与原因归档 |

## 二、 6×5 生态位饱和度矩阵 (Cluster A~F × Timeframe `1d`/`60m`/`30m`/`15m`/`5m`)

> 单元格格式：`Gate1B通过折数 / 评估UID数`（括号内为平均 Sharpe；`🟢 SCARCE 蓝海`、`🟡 BALANCED 适中`、`🔴 SATURATED 饱和`）。

| 机制主簇 (Cluster) | `1d` | `60m` | `30m` | `15m` | `5m` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A · Trend / Regime (趋势与宏观状态)** | 🟢 `0/11` (-0.23) | 🟢 `1/11` (-0.22) | 🟡 `5/13` (-0.21) | 🟢 `0/4` (-0.54) | 🟢 `0/2` (-0.97) |
| **B · Breakout / Participation (突破与资金参与)** | 🟡 `4/10` (-0.00) | 🟢 `0/6` (-0.14) | 🟢 `0/6` (-0.18) | 🟢 `0/6` (-0.27) | 🟢 `0/4` (-0.35) |
| **C · Pullback / Exhaustion (回调与衰竭反转)** | 🟡 `4/12` (-0.02) | 🟢 `1/8` (-0.21) | 🟢 `0/8` (-0.33) | 🟢 `1/6` (-0.68) | 🟢 `0/2` (-0.05) |
| **D · Position / Carry / Basis (持仓博弈与期限结构)** | 🟢 `0/10` (-0.21) | 🟢 `1/10` (-0.24) | 🟢 `1/8` (-0.39) | 🟢 `0/4` (-0.70) | 🟢 `0/2` (-0.09) |
| **E · Cross-Asset / Relative Value (截面强弱与相对价值)** | 🟢 `0/8` (-0.20) | 🟡 `6/10` (-0.07) | 🟢 `1/8` (-0.25) | 🟢 `2/8` (-0.40) | 🟢 `0/0` |
| **F · Intraday / Execution Microstructure (日内微结构与冲击)** | 🟢 `0/6` (+0.00) | 🟢 `1/8` (-0.25) | 🟢 `1/8` (-0.27) | 🟢 `1/8` (-0.49) | 🟢 `0/0` |

## 三、 Stage 1 ~ Stage 4 全周期组合消融演进对比 (`ablation_runs`)

| 运行标识 (`run_id`) | 阶段 (`stage`) | 模式 (`mode`) | Flat-60 年化 | Flat-60 Sharpe | Flat-60 Calmar | 机构两层年化 | 机构两层 Sharpe | 机构最大回撤 | 机构 Calmar | 单品种均因子数 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Stage1_Base_Full60` | `Stage1_Base` | `baseline` | `+2.27%` | **+1.150** | `+1.200` | `+4.12%` | **+1.880** | `1.64%` | `+2.510` | `6.8` |
| `Full60_FlatHRP_FullStructExit` | `Stage2_Rust` | `architecture_variant` | `+1.23%` | **+0.610** | `+0.470` | `+1.23%` | **+0.610** | `2.63%` | `+0.470` | `7.2` |
| `Full60_FlatHRP_FullStructExit_3x` | `Stage2_Rust` | `3x_target` | `+3.70%` | **+0.610** | `+0.510` | `+3.70%` | **+0.610** | `7.24%` | `+0.510` | `7.2` |
| `Full60_FlatHRP_HysteresisRunner` | `Stage2_Rust` | `architecture_variant` | `+2.27%` | **+1.150** | `+1.200` | `+2.27%` | **+1.150** | `1.88%` | `+1.200` | `7.2` |
| `Full60_FlatHRP_HysteresisRunner_3x` | `Stage2_Rust` | `3x_target` | `+6.80%` | **+1.150** | `+1.260` | `+6.80%` | **+1.150** | `5.41%` | `+1.260` | `7.2` |
| `Full60_FlatHRP_StaticExit` | `Stage2_Rust` | `architecture_variant` | `+2.16%` | **+1.030** | `+1.070` | `+2.16%` | **+1.030** | `2.02%` | `+1.070` | `7.2` |
| `Full60_FlatHRP_StaticExit_3x` | `Stage2_Rust` | `3x_target` | `+6.48%` | **+1.030** | `+1.150` | `+6.48%` | **+1.030** | `5.65%` | `+1.150` | `7.2` |
| `Full60_SecConserved_SymDZ_HR` | `Stage2_Rust` | `architecture_variant` | `+2.61%` | **+1.210** | `+1.350` | `+2.61%` | **+1.210** | `1.93%` | `+1.350` | `7.2` |
| `Full60_SecConserved_SymDZ_HR_3x` | `Stage2_Rust` | `3x_target` | `+7.83%` | **+1.210** | `+1.520` | `+7.83%` | **+1.210** | `5.14%` | `+1.520` | `7.2` |
| `Full60_ZeroHardcode_ClusterConserved_3x` | `Stage2_Rust` | `3x_target` | `+12.37%` | **+1.860** | `+2.450` | `+12.37%` | **+1.860** | `5.05%` | `+2.450` | `7.2` |
| `Full60_ZeroHardcode_ClusterConserved_Optimal` | `Stage2_Rust` | `architecture_variant` | `+4.12%` | **+1.860** | `+2.440` | `+4.12%` | **+1.860** | `1.69%` | `+2.440` | `7.2` |
| `Full60_ZeroHardcode_SOPStage3` | `Stage2_Rust` | `architecture_variant` | `+4.12%` | **+1.880** | `+2.510` | `+4.12%` | **+1.880** | `1.64%` | `+2.510` | `7.2` |
| `Full60_ZeroHardcode_SOPStage3_3x` | `Stage2_Rust` | `3x_target` | `+12.35%` | **+1.880** | `+2.520` | `+12.35%` | **+1.880** | `4.89%` | `+2.520` | `7.2` |
| `Stage2_Rust_Incremental` | `Stage2_Rust` | `incremental` | `+2.61%` | **+1.210** | `+1.350` | `+4.12%` | **+1.860** | `1.69%` | `+2.440` | `7.2` |
| `Stage3_OI_Incremental` | `Stage3_OI` | `incremental` | `+2.64%` | **+1.220** | `+1.380` | `+4.15%` | **+1.890** | `1.63%` | `+2.550` | `7.4` |
| `Stage3_OI_Standalone` | `Stage3_OI` | `standalone` | `+0.84%` | **+0.520** | `+0.390` | `+1.42%` | **+0.780** | `1.95%` | `+0.730` | `1.6` |
| `Stage4_CE_Incremental` | `Stage4_CE` | `incremental` | `+2.76%` | **+1.280** | `+1.490` | `+4.28%` | **+1.940** | `1.61%` | `+2.660` | `7.9` |
| `Stage4_CE_Standalone` | `Stage4_CE` | `standalone` | `+1.38%` | **+0.890** | `+0.780` | `+2.19%` | **+1.240** | `1.68%` | `+1.300` | `1.9` |

## 四、 各主簇 (Cluster A~F) Top 准入因子排行榜 (`factor_pool_scores`)

| 主簇 | 排名 | 候选 UID (`uid`) | 家族 (`family_id`) | 周期 | 阶段 (`stage`) | 检验窗口 (`fold`) | 全池 Mean Sharpe | Pooled-t | 正收益品种占比 | Gate1B 准入 |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 1 | `H30M_AC04L@30m` | `H30M_AC04` | `30m` | `Stage2_Rust` | `Fold1_2025` | **+0.1820** | `+0.12` | `56.7%` | ✅ |
| **A** | 2 | `S1S@30m` | `S1S` | `30m` | `Stage1_Base` | `Fold2_2026` | **-0.0940** | `-1.51` | `48.3%` | ✅ |
| **A** | 3 | `S1L@60m` | `S1L` | `60m` | `Stage1_Base` | `Fold2_2026` | **-0.1760** | `-2.63` | `40.0%` | ✅ |
| **A** | 4 | `H30M_AC04S@30m` | `H30M_AC04` | `30m` | `Stage2_Rust` | `Fold2_2026` | **-0.1850** | `-2.29` | `43.3%` | ✅ |
| **A** | 5 | `S1L@30m` | `S1L` | `30m` | `Stage1_Base` | `Fold2_2026` | **-0.2330** | `-2.85` | `38.3%` | ✅ |
| **B** | 1 | `H1D_BC01S@1d` | `H1D_BC01` | `1d` | `Stage2_Rust` | `Fold1_2025` | **+0.2660** | `+3.19` | `61.7%` | ✅ |
| **B** | 2 | `H1D_DV01S@1d` | `H1D_DV01` | `1d` | `Stage1_Base` | `Fold1_2025` | **+0.2370** | `+2.90` | `51.7%` | ✅ |
| **B** | 3 | `BC01S@60m` | `BC01` | `60m` | `Stage1_Base` | `Fold2_2026` | **+0.1330** | `-0.80` | `61.7%` | ➖ |
| **B** | 4 | `BC04L@1d` | `BC04` | `1d` | `Stage3_OI` | `Fold2_2026` | **+0.0800** | `+1.36` | `55.0%` | ➖ |
| **B** | 5 | `H5M_LQ01S@5m` | `H5M_LQ01` | `5m` | `Stage1_Base` | `Fold1_2025` | **+0.0580** | `+1.10` | `50.0%` | ➖ |
| **C** | 1 | `H1D_DZ01L@1d` | `H1D_DZ01` | `1d` | `Stage2_Rust` | `Fold1_2025` | **+0.3070** | `+4.12` | `61.7%` | ✅ |
| **C** | 2 | `H1D_DZ01S@1d` | `H1D_DZ01` | `1d` | `Stage2_Rust` | `Fold2_2026` | **+0.0910** | `+5.68` | `51.7%` | ✅ |
| **C** | 3 | `H1D_SP01S@1d` | `H1D_SP01` | `1d` | `Stage1_Base` | `Fold2_2026` | **+0.0440** | `+5.82` | `51.7%` | ✅ |
| **C** | 4 | `CC01S@60m` | `CC01` | `60m` | `Stage1_Base` | `Fold2_2026` | **-0.1070** | `-1.21` | `43.3%` | ✅ |
| **C** | 5 | `H15M_PB01L@15m` | `H15M_PB01` | `15m` | `Stage1_Base` | `Fold2_2026` | **-0.1900** | `-2.03` | `38.3%` | ✅ |
| **D** | 1 | `DC02L@60m` | `DC02` | `60m` | `Stage1_Base` | `Fold2_2026` | **-0.2470** | `-4.26` | `36.7%` | ✅ |
| **D** | 2 | `DC02L@30m` | `DC02` | `30m` | `Stage1_Base` | `Fold2_2026` | **-0.3470** | `-4.93` | `28.3%` | ✅ |
| **D** | 3 | `H1D_RV01S@1d` | `H1D_RV01` | `1d` | `Stage1_Base` | `Fold1_2025` | **+0.1540** | `+1.66` | `55.0%` | ➖ |
| **D** | 4 | `H60M_DC03S@60m` | `H60M_DC03` | `60m` | `Stage2_Rust` | `Fold1_2025` | **+0.1270** | `+0.92` | `51.7%` | ➖ |
| **D** | 5 | `H1D_RV01L@1d` | `H1D_RV01` | `1d` | `Stage1_Base` | `Fold1_2025` | **+0.0260** | `+0.23` | `46.7%` | ➖ |
| **E** | 1 | `EC05L@60m` | `EC05` | `60m` | `Stage4_CE` | `Fold2_2026` | **+0.0910** | `+0.73` | `61.7%` | ✅ |
| **E** | 2 | `EC02S@60m` | `EC02` | `60m` | `Stage1_Base` | `Fold2_2026` | **+0.0010** | `-0.34` | `55.0%` | ✅ |
| **E** | 3 | `H15M_CL01L@15m` | `H15M_CL01` | `15m` | `Stage1_Base` | `Fold2_2026` | **-0.1130** | `-2.22` | `48.3%` | ✅ |
| **E** | 4 | `H15M_TN01L@15m` | `H15M_TN01` | `15m` | `Stage2_Rust` | `Fold2_2026` | **-0.1160** | `-1.81` | `35.0%` | ✅ |
| **E** | 5 | `EC02L@60m` | `EC02` | `60m` | `Stage1_Base` | `Fold2_2026` | **-0.1240** | `-1.74` | `46.7%` | ✅ |
| **F** | 1 | `FC03S@15m` | `FC03` | `15m` | `Stage1_Base` | `Fold2_2026` | **-0.2160** | `-3.00` | `36.7%` | ✅ |
| **F** | 2 | `FC01L@30m` | `FC01` | `30m` | `Stage1_Base` | `Fold2_2026` | **-0.3090** | `-3.19` | `33.3%` | ✅ |
| **F** | 3 | `FC02L@60m` | `FC02` | `60m` | `Stage1_Base` | `Fold1_2025` | **-0.4070** | `-4.03` | `35.0%` | ✅ |
| **F** | 4 | `FC03L@60m` | `FC03` | `60m` | `Stage1_Base` | `Fold2_2026` | **+0.0060** | `-0.88` | `51.7%` | ➖ |
| **F** | 5 | `FC03L@1d` | `FC03` | `1d` | `Stage1_Base` | `Fold1_2025` | **+0.0000** | `+0.00` | `0.0%` | ➖ |

## 五、 28 个 SSOT 原子因子族 AST 静态血缘与依赖矩阵 (`catalog.py`)

| 家族 (`family_id`) | 主簇 | 阶段 (`stage`) | 默认周期 | State 依赖数 | Strength 依赖 | OI 依赖 | CS 截面依赖 | HTF 依赖 | 代码指纹 (`code_sha256[:10]`) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AC01` | **A** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `zs_break` | `-` | `-` | `-` | `f8499d326e` |
| `AC02` | **A** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `1` | `-` | `-` | `-` | `-` | `b37997bec2` |
| `AC03` | **A** | `Stage2_Rust` | `1d,60m,30m,15m,5m` | `5` | `-` | `-` | `-` | `-` | `ad9444b1d2` |
| `AC04` | **A** | `Stage2_Rust` | `1d,60m,30m,15m,5m` | `4` | `zs_break` | `-` | `-` | `-` | `538a58020c` |
| `AC05` | **A** | `Stage3_OI` | `1d,60m,30m,15m,5m` | `5` | `-` | `oi_flow3_z/oi_ratio` | `-` | `-` | `100131657c` |
| `BC01` | **B** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `bi_div/fx` | `-` | `-` | `-` | `f8e49b5da1` |
| `BC02` | **B** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `2` | `-` | `-` | `-` | `-` | `3116bb41a5` |
| `BC03` | **B** | `Stage2_Rust` | `1d,60m,30m,15m,5m` | `4` | `bi_div/fx` | `-` | `-` | `-` | `04a5c08b77` |
| `BC04` | **B** | `Stage3_OI` | `1d,60m,30m,15m,5m` | `4` | `bi_div/fx` | `oi_chg_z/oi_flow3_z` | `-` | `-` | `99faedc31b` |
| `CC01` | **C** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `zs_revert` | `-` | `-` | `-` | `103e33514c` |
| `CC02` | **C** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `-` | `-` | `-` | `-` | `b11751572e` |
| `CC03` | **C** | `Stage2_Rust` | `1d,60m,30m,15m,5m` | `3` | `zs_revert` | `-` | `-` | `-` | `807e087306` |
| `CC04` | **C** | `Stage4_CE` | `1d,60m,30m,15m,5m` | `11` | `flow/zs_revert` | `oi_chg_z/oi_flow3_z` | `-` | `-` | `68ad4d1553` |
| `CC05` | **C** | `Stage4_CE` | `1d,60m,30m,15m,5m` | `11` | `fx/zs_revert` | `oi_flow3_z` | `-` | `-` | `c33688b8c9` |
| `DC01` | **D** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `1` | `flow` | `-` | `-` | `-` | `9732f4570f` |
| `DC02` | **D** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `2` | `flow` | `-` | `-` | `-` | `cb9504d98b` |
| `DC03` | **D** | `Stage2_Rust` | `1d,60m,30m,15m,5m` | `4` | `flow` | `-` | `-` | `-` | `397307441f` |
| `DC04` | **D** | `Stage3_OI` | `1d,60m,30m,15m,5m` | `7` | `flow` | `oi_avail/oi_chg_z/oi_flow3_z` | `-` | `-` | `5b3990d627` |
| `DC05` | **D** | `Stage3_OI` | `1d,60m,30m,15m,5m` | `7` | `flow` | `oi_chg_z/oi_flow3_z` | `-` | `-` | `c7780411f6` |
| `EC01` | **E** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `squeeze` | `-` | `-` | `-` | `f50248c34c` |
| `EC02` | **E** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `-` | `-` | `-` | `-` | `e37312dc37` |
| `EC03` | **E** | `Stage2_Rust` | `1d,60m,30m,15m,5m` | `4` | `squeeze` | `-` | `-` | `-` | `814c111f28` |
| `EC04` | **E** | `Stage4_CE` | `1d,60m,30m,15m,5m` | `9` | `squeeze/zs_break` | `oi_flow3_z` | `-` | `-` | `7146ab2830` |
| `EC05` | **E** | `Stage4_CE` | `1d,60m,30m,15m,5m` | `8` | `squeeze` | `oi_flow3_z` | `-` | `-` | `3c3aeb81ab` |
| `FC01` | **F** | `Stage1_Base` | `60m,30m,15m,5m` | `0` | `fx` | `-` | `has_external_cs/xs` | `htf_dir/htf_q` | `cd59439436` |
| `FC02` | **F** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `fx` | `-` | `leader_lag_gap/sector_leader_dir` | `-` | `b241ab0cbf` |
| `FC03` | **F** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `bi_div/fx/zs_revert` | `-` | `sector_spread_zscore` | `-` | `a4d70085be` |
| `FC04` | **F** | `Stage1_Base` | `1d,60m,30m,15m,5m` | `0` | `zs_break` | `-` | `cs_strength_eff` | `-` | `2c537ac170` |

## 六、 待开采高优先级 Rust 原生信号推荐池 (`exact_mirror=True` & `consumed_count=0`)

> 共发现 **118** 个已通过仿射反转严格镜像对称验证、且尚未接入现有原子因子族的原生 Rust 信号。以下按信号分类展示代表性蓝海信号：

| 信号子模块 (`category`) | 待开采对称信号数 | 代表性未开采 `exact_mirror=True` 原生信号示例 |
| :--- | :---: | :--- |
| `ang` | `7` | `amv_up_dw_line_V230603`, `asi_up_dw_line_V230603`, `dema_up_dw_line_V230605`, `er_up_dw_line_V230604` |
| `bar` | `28` | `bar_bpm_V230227`, `bar_classify_V240606`, `bar_classify_V240607`, `bar_intraday_direction_V260424` |
| `byi` | `1` | `byi_fx_num_V230628` |
| `coo` | `2` | `coo_cci_V230323`, `coo_td_V221110` |
| `cxt` | `21` | `cxt_bi_end_V230815`, `cxt_decision_V240526`, `cxt_fx_power_V221107`, `cxt_nine_bi_V230621` |
| `jcc` | `11` | `jcc_shan_chun_V221121`, `jcc_ping_tou_V221113`, `jcc_two_crow_V221108`, `jcc_szx_V221111` |
| `obv` | `1` | `obv_up_dw_line_V230719` |
| `pressure` | `1` | `pressure_support_V240406` |
| `tas` | `36` | `tas_angle_V230802`, `tas_ma_round_V221206`, `tas_macd_base_V221028`, `tas_macd_base_V230320` |
| `vol` | `3` | `vol_window_V230731`, `vol_double_ma_V230214`, `vol_single_ma_V230214` |
| `xl` | `3` | `xl_bar_position_V240328`, `xl_bar_trend_V240330`, `xl_bar_basis_V240411` |
| `zdy` | `4` | `zdy_dif_V230527`, `zdy_macd_V230527`, `zdy_macd_dif_iqr_V230521`, `zdy_bi_end_V230406` |

## 七、 失败实验墓地统计与典型淘汰案例 (`factor_graveyard`)

| 淘汰门禁 (`failed_gate`) | 累计淘汰候选数 | 门禁拦截规则说明 |
| :--- | :---: | :--- |
| `Gate2_ColinearClone` | `9` | 同簇激活条件共线克隆 (|ρ| >= 0.80)，缺乏独立机制增量 |
| `Gate2_HighActiveCorr` | `20` | 与已入库因子在活跃区高度正交冗余 (|ρ_active| >= 0.35) |
| `Gate2_Weak1DPoolEdge` | `21` | 1d 日线跨品种池化稳健度或扣费后净夏普未达准入线 |
| `Gate2_Weak5MPoolEdge` | `3` | 5m 高频跨品种池化净边际不足以覆盖冲击与手续费损耗 |
| `Gate2_WeakIntradayPoolEdge` | `4` | 日内周期跨品种池化 t 统计量或正收益品种占比未达标 |
| `Gate2_WeakPoolEdge` | `31` | 15m/30m 跨品种普适性不足 (Pooled-t < 1.96 或 PosRatio < 50%) |
| `Gate3_NegativeMarginalSharpe` | `7` | 加入正交组合后扣费边际净夏普无正增益 (ΔSharpe <= 0) |

### 最近归档的 Top 12 失败/冗余候选明细

| 候选标识 (`uid`) | 家族 (`family_id`) | 周期 | 阶段 | 淘汰门禁 | Mean IS Sharpe | 年化换手 | 淘汰原因摘要 |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| `AC01L@15m` | `AC01` | `15m` | `Stage1_Base` | `Gate2_WeakPoolEdge` | `+0.8180` | `65.98` | 15m跨品种普适性不足 (pooled_t=-1.46, pos_ratio=0.40) |
| `AC01L@1d` | `AC01` | `1d` | `Stage1_Base` | `Gate2_Weak1DPoolEdge` | `+1.1260` | `5.75` | 1d池化稳健度或本地夏普不足 (pooled_t=-1.44, Shp=1.19) |
| `AC01L@60m` | `AC01` | `60m` | `Stage1_Base` | `Gate2_HighActiveCorr` | `+0.8600` | `15.85` | 与已入库 AC01S@30m 激活态共线冗余 (|ρ_active|=0.47 >= 0.35) |
| `AC01S@15m` | `AC01` | `15m` | `Stage1_Base` | `Gate2_WeakPoolEdge` | `+0.4220` | `56.90` | 15m跨品种普适性不足 (pooled_t=-2.73, pos_ratio=0.40) |
| `AC01S@60m` | `AC01` | `60m` | `Stage1_Base` | `Gate2_ColinearClone` | `+0.4720` | `12.80` | 同簇激活条件相关过高 (|ρ|=0.86>=0.80 vs S1L@60m) |
| `AC02L@15m` | `AC02` | `15m` | `Stage1_Base` | `Gate2_WeakPoolEdge` | `+0.8360` | `70.45` | 15m跨品种普适性不足 (pooled_t=-2.38, pos_ratio=0.30) |
| `AC02L@1d` | `AC02` | `1d` | `Stage1_Base` | `Gate2_Weak1DPoolEdge` | `+0.9490` | `5.00` | 1d池化稳健度或本地夏普不足 (pooled_t=-4.56, Shp=1.34) |
| `AC02L@60m` | `AC02` | `60m` | `Stage1_Base` | `Gate3_NegativeMarginalSharpe` | `+0.7040` | `12.50` | 扣费后边际净夏普无正增益 (ΔSharpe=-0.0194 < 0, 换手=8.9x) |
| `AC02S@15m` | `AC02` | `15m` | `Stage1_Base` | `Gate2_WeakPoolEdge` | `+0.7190` | `58.73` | 15m跨品种普适性不足 (pooled_t=-1.40, pos_ratio=0.35) |
| `AC02S@60m` | `AC02` | `60m` | `Stage1_Base` | `Gate2_HighActiveCorr` | `+0.9040` | `11.90` | 与已入库 AC02L@60m 激活态共线冗余 (|ρ_active|=0.58 >= 0.35) |
| `BC01L@15m` | `BC01` | `15m` | `Stage1_Base` | `Gate2_WeakPoolEdge` | `+0.7180` | `24.28` | 15m跨品种普适性不足 (pooled_t=-2.83, pos_ratio=0.35) |
| `BC01L@60m` | `BC01` | `60m` | `Stage1_Base` | `Gate3_NegativeMarginalSharpe` | `+1.7260` | `6.50` | 扣费后边际净夏普无正增益 (ΔSharpe=-0.0068 < 0, 换手=6.5x) |
