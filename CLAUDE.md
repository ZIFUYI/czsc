# CLAUDE.md

本文件是 Codex、Claude Code 等编码 agent 的仓库工作指引。根目录 `AGENTS.md` 是指向本文件的软链接；更新指引时只修改 `CLAUDE.md`，保持软链接。

目录、依赖、公共 API 和 CI 命令分别以实际源码、`Cargo.toml` / `pyproject.toml`、`czsc/__init__.py` 和 `.github/workflows/` 为准。发现历史实现与下述开发宪法冲突时，应修复实现，不得据此放宽硬约束。

## 项目概述

CZSC（缠中说禅技术分析工具）是基于缠中说禅理论的综合性量化交易Python库，提供技术分析、信号生成、回测和市场分析等功能。本项目专注于实现缠论的分型、笔、线段等核心概念的自动识别，以及基于此的多级别量化交易策略。

## 🏛️ 开发宪法（Constitution）

以下规则是本项目长期演进的**硬约束**，任何 PR、任何重构、任何"为了赶进度的临时变通"都不得违反。与此冲突的代码审查意见、个人偏好、历史代码一律以本节为准；违反本节的代码即便已合入，也按 bug 处理、必须回滚或修复。

### 第一条 · Rust ↔ Python 行为一致

**需要 Rust 实现的部分必须同时满足 Rust crate 与 Python wheel 行为一致（Python 端纯透传，禁止再写适配层）。**

> 落地参考：策略信号去重与单个 Position 的保存、加载及完整性校验由 `crates/czsc-trader/src/strategy.rs` 实现，Python 策略门面调用对应的 `_native.strategy_*` API。文件校验使用 `sha256(canonical JSON)`，可在 Rust / Python 端 byte-for-byte 一致复现。回归验证见 `tests/unit/test_strategy_unique_signals_parity.py` 和 `tests/unit/test_strategy_save_load_parity.py`。

具体含义：

- 同一个名字（如 `monotonicity` / `CzscStrategyBase` / `generate_czsc_signals` / `CZSC`），`cargo add czsc` 的 Rust 用户与 `pip install czsc` 的 Python 用户调用后**行为必须一致**——同样的输入产生同样的输出，默认参数、错误处理、边界条件、字段命名都一致。
- Python 侧**只允许**做下面两类工作：
  1. **纯透传**：`from czsc._native import xxx` 后直接 re-export，不做任何包装；
  2. **不可避免的 PyO3 边界胶水**：DataFrame ↔ Arrow IPC 序列化、`pathlib.Path` ↔ `String` 转换等，PyO3 类型系统无法跨越的边界处理。
- **禁止**在 Python 侧做参数归一化、默认值补齐、返回值字段重命名、错误码翻译、`isinstance` 多态分支等"适配层"工作——这类逻辑必须下沉到 Rust 端实现（修改现有 API 或新增 API）。
- 新增 Python wrapper 之前，PR 描述里必须先回答"为什么不能改成 Rust 实现"，并经过 reviewer 显式批准。

**违反本条的常见信号（在 review 中视为红线）**：

- Python 函数体内出现 `if isinstance(bars, pd.DataFrame): ... elif isinstance(bars, list): ...` 等多态分支；
- Python 函数返回的 dict 字段顺序 / 命名与 Rust 端 `serde` 输出不一致；
- 同一份功能在 Python 测试覆盖完整，但 `cargo test` 没有等价用例；
- CHANGELOG 写 "Python 端默认参数从 X 改为 Y"，但 Rust 端无对应改动；
- `czsc/_runtime_adapters.py` 等"适配层"文件持续膨胀，而不是被逐步搬空到 Rust。

## 常用开发命令

### Git 提交约定

- 需要提交并推送到远程仓库时，默认直接提交到主干 `master` 并推送 `origin/master`；不要创建功能分支或 PR 分支，除非用户明确要求。
- 提交前只暂存本次任务相关文件，避免把工作区已有的无关改动带入提交。
- 开工前检查 `git status --short --branch`，保留已有改动和未跟踪文件。

### UV 包管理 (项目使用UV管理依赖)

```bash
# 首次建立环境，或 pyproject.toml / uv.lock 变更后同步
# 默认安装 dependency-groups.dev；test extra 包含 pytest-cov 等测试依赖
uv sync --extra test

# 需要全部可选依赖时使用
uv sync --extra all

# 首次构建，或修改 Rust 源码 / Cargo 配置后，重建 Python 扩展
uv run --no-sync maturin develop --release

# 运行测试（日常默认 --no-sync，省去每次 4-5s 的 lockfile/venv 一致性检查）
uv run --no-sync pytest

# 运行指定测试文件
uv run --no-sync pytest tests/unit/test_structure_analysis_api.py -v

# 运行单个测试函数
uv run --no-sync pytest tests/unit/test_structure_analysis_api.py::test_czsc_exposes_zs_list_property -v

# 带覆盖率的测试
uv run --no-sync pytest --cov=czsc

# 跑全套（含 @pytest.mark.slow 标记的耗时测试，CI / 发布前用）
uv run --no-sync pytest --run-slow

# 格式化时只对本次修改的文件运行 ruff format；以下检查不改写文件
uv run --no-sync ruff format --check czsc/ tests/
uv run --no-sync ruff check czsc/ tests/
```

> **`--no-sync` 约定**：日常开发使用已同步的环境，统一加 `--no-sync`；首次建立环境或 `pyproject.toml` / `uv.lock` 变更后显式跑 `uv sync`。修改 Rust 后必须先重建 `czsc._native` 再跑 Python 测试，不能用已有扩展验证新源码。项目使用 Ruff，不使用 Black / isort / flake8。

### Rust 检查与类型 stub

```bash
# 使用项目 Python（涉及 PyO3 链接时需要 Python >= 3.10）
export PYO3_PYTHON="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"

# 按本次修改的 crate 跑 Rust 测试；此处以核心算法为例
cargo test -p czsc-core --no-fail-fast
# 修改信号或交易逻辑时，分别使用以下测试入口
cargo test -p czsc-signals --no-fail-fast
cargo test -p czsc-trader --no-fail-fast

# 与 CI 一致的格式和 lint 检查
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# 修改 PyO3 公共 API 或 gen_stub_* 注册后重新生成 stub
cargo run --bin stub_gen -p czsc-python --no-default-features --features stub-gen
git diff -- czsc/_native/__init__.pyi
```

- `czsc-python` 默认启用 `extension-module` 和 `abi3-py310`。不要直接以 `cargo test --workspace` 代替逐 crate 测试：feature 合并可能使测试二进制无法链接 libpython。
- CI 的 Rust 测试当前覆盖 `czsc-derive / czsc-core / czsc-utils / czsc-ta / czsc-signal-macros`；信号和交易逻辑修改还需运行对应 crate 的测试及 Python 回归测试。Python 矩阵覆盖 3.10–3.13，完整命令见 `.github/workflows/code-quality.yml`。
- stub 生成必须同时传 `--no-default-features --features stub-gen`，以启用生成器并关闭扩展模块的链接方式。提交生成结果；CI 会重新生成并检查漂移。
- `czsc/_native/__init__.pyi` 由生成器维护，不手改，也不交给 Ruff 格式化。其余 Python stub 修改需与对应公共接口同步。

### 测试规范

- Python 测试位于 `tests/`，使用 pytest；Rust 单元测试及集成测试位于对应 crate 的源码和 `tests/` 目录。
- **关键原则**：测试数据统一通过 `czsc.mock` 模块获取，不要在测试中硬编码模拟数据
- 测试文件命名模式：`test_*.py`
- 模拟数据使用 `generate_symbol_kines` 函数生成，支持多品种、多频率、可重现的随机数据
- **慢测试约定**：依赖 `time.sleep` 或子进程冷启动的测试加 `@pytest.mark.slow`，默认跳过；CI / 发布前用 `pytest --run-slow` 跑全套。注册逻辑见 `tests/conftest.py`。
- 根据改动选择验证范围：核心分析看 `tests/unit/`，公共 API / stub 看 `tests/compat/`，命令行看 `tests/cli/`，可视化看 `tests/test_lightweight*.py`。涉及 Rust/Python 共享行为时，两端都必须有等价用例。


## 代码架构

### 核心组件

1. **`czsc._native`** - PyO3 编译产生的 Rust 扩展模块（缠论核心）：
   - 由 `crates/czsc-python` 通过 `maturin` 打包，扩展模块名 `czsc._native`
   - 暴露 `CZSC / FX / BI / ZS / RawBar / NewBar / Freq / Mark / Direction / Operate / Signal / Event / Position / BarGenerator` 等核心类型
   - 暴露 `check_bi / check_fx / check_fxs / remove_include / freq_end_time / is_trading_time` 等工具函数
   - 结构分析公共 API 包括 `create_fake_bis / get_zs_seq / is_symmetry_zs / is_bis_up / is_bis_down / check_gap_info`，以及 `CZSC.zs_list`；共享实现位于 `crates/czsc-core/`，Python 使用顶层 `czsc.*`
   - 提供 Rust 信号注册表和统一分发器；信号发现与调用入口见下述信号模块
   - 暴露 `czsc._native.ta.*`（Rust TA 算子，供信号函数内部使用）；Python 不提供 `czsc.ta` 顶层 alias
   - **不存在 Python 回退**：核心分析统一由 Rust 实现。

2. **`crates/`** - Rust workspace（9 个 crate）：
   - `czsc` / `czsc-core` / `czsc-derive` / `czsc-signals` / `czsc-trader` / `czsc-utils` / `czsc-ta`
   - `czsc-signal-macros`（proc-macro，`#[signal]` 注册宏）
   - `czsc-python`（PyO3 binding 总入口，唯一启用 `pyo3/extension-module` 的 crate）

3. **`czsc/traders/`** - 交易执行框架：
   - `__init__.py`：facade，统一 re-export `CzscSignals / CzscTrader / generate_czsc_signals / get_signals_config / get_signals_freqs / derive_signals_config / derive_signals_freqs / get_unique_signals / WeightBacktest`，全部来自 `czsc._native` 或 `wbt`
   - 用户代码通过 `czsc.*` 或 `czsc.traders` 调用；内部绑定透传使用 `czsc._native`。已删除的 `base.py / sig_parse.py` 不再作为入口
   - 优化工具位于 `czsc/utils/optimize.py`：`OpensOptimize / ExitsOptimize / CzscOpenOptimStrategy / CzscExitOptimStrategy`

4. **`czsc._native.signals`** - Rust 信号分发器的内部 Python 命名空间：
   - 当前注册 7 个分类子模块（bar/cvolp/cxt/obv/pressure/tas/vol）；分类注册以 `crates/czsc-python/src/signals_dispatcher.rs` 为准，不等于 Rust 信号模块的完整清单
   - Rust 模块以 `crates/czsc-signals/src/lib.rs` 为准，可用 `rg --files crates/czsc-signals/src -g '*.rs'` 查看源码；不要依赖固定文件数或信号数量
   - 用户通过 `uv run --no-sync czsc signals list --json` 查询信号目录，通过 `czsc.get_signals_config / czsc.get_signals_freqs` 解析配置，通过 `czsc.generate_czsc_signals` 等接口计算信号
   - 原 `czsc/signals/` 已删除，不恢复历史 Python 信号实现；依赖 trader 状态的信号通过交易器更新路径分发

5. **`czsc/utils/`** - 工具模块：
   - `data/cache.py` / `io.py` / `log.py` / `kline_quality.py`：缓存、IO、日志、K 线质量校验
   - `analysis/corr.py`：`cross_sectional_ic`；`daily_performance / top_drawdowns` 由顶层 `czsc.*` 透传 wbt
   - `data/client.py`：统一数据客户端接口
   - TA 算子由 Rust `czsc._native.ta` 提供（信号内部依赖），顶层别名 `czsc.{ema,sma,rolling_rank,boll_positions,ultimate_smoother}` 保留；仪表盘场景的 MACD（×2 约定）已下沉为 `czsc/utils/plotting/_macd.py` 私有辅助
   - `trade.py`：交易工具
   - `plotting/{kline,weight}.py` + `plotting/lightweight/`：plotly 单周期 K 线 + 权重时序图 / lightweight-charts 自包含 HTML
   - 历史 Streamlit 组件、报表工具、`plotting/backtest.py` 等已删除，不恢复这些入口；迁移清单见 `docs/migration/cleanup-non-czsc-core.md`

6. **`czsc/cli/`** - Typer 命令行入口：
   - `pyproject.toml [project.scripts]` 注册 `czsc = "czsc.cli:app"`
   - 命令包括 `signals / research / data / plot / analyze / backtest / bench / schema`，支持 JSON 输出；命令参数以 `--help` 和 `czsc schema --json` 为准
   - 参数与输出契约修改需同步 `tests/cli/`；业务算法继续复用 Rust 后端，遵循开发宪法

7. **`czsc/connectors/`** - 数据源连接器：
   - 支持天勤、Tushare、CCXT、聚宽等数据源
   - 统一的数据接口封装；`local_data.py`（原 `research.py`）提供 CZSC 投研共享数据的本地缓存读取入口

### 信号-事件-交易体系

项目实现了系统化的量化交易方法：
- **信号（Signals）**: 基础技术指标和市场状态
- **事件（Events）**: 信号的逻辑组合，通过 signals_all/signals_any/signals_not 实现 AND/OR/NOT 逻辑
- **交易（Trading）**: 基于事件和风险管理的执行

### 多级别联立分析

CZSC 支持使用 `CzscTrader` 类进行多级别联立分析，可同时分析不同时间周期（如1分钟、5分钟、30分钟、日线）进行全面的市场决策。

## 开发指南

### 代码规范
- 行长度：120字符（在 pyproject.toml 中配置）
- 适当使用类型提示
- 遵循代码库中现有的命名约定
- 信号函数通过 `#[signal]` 宏在 Rust 端自动注册到 `SIGNAL_REGISTRY`，命名遵循 Rust 模块约定，不再使用历史上的 `V<yyMMdd>` 版本后缀
- **代码质量原则**：
  - **DRY（Don't Repeat Yourself）**: 提取重复代码为辅助函数
  - **KISS（Keep It Simple）**: 保持函数简洁，职责单一
  - **使用模块级常量**: 避免魔法值，集中管理配置
  - **类型提示优先**: 使用 `Literal`、`Optional` 等提升代码可读性
  - **向后兼容性**: 公共 API 修改需谨慎，避免破坏现有代码
  - **文档完整**: 所有公共函数必须有完整的 docstring

### 信号函数开发

- 信号函数应遵循飞书文档中的规范说明
- 信号函数由 Rust 实现，位于 `crates/czsc-signals/` 中
- 用 `#[signal]` 注册新信号，先通过 `uv run --no-sync czsc signals list --json` 检查现有名称和参数模板，再通过 `czsc.get_signals_config(signals_seq)` / `czsc.get_signals_freqs(signals_seq)` 解析配置；不要臆造信号字符串
- 注册表与源码清单见「核心组件」的信号模块说明；新增信号需覆盖 Rust 行为及 Python 公共调用路径

### 数据处理最佳实践
- 测试数据统一通过 `czsc.mock.generate_symbol_kines` 生成
- 使用 `format_standard_kline` 将DataFrame转换为RawBar对象列表
- 使用 `BarGenerator` 进行K线合成和多级别分析
- 通过 `DataClient` 统一访问不同数据源
- 注意使用磁盘缓存提高重复计算效率

### 数据格式转换
```python
# 从 mock 数据生成 CZSC 对象（全部走顶层 czsc 命名空间）
from czsc import CZSC, Freq, format_standard_kline
from czsc.mock import generate_symbol_kines

# 生成K线数据
df = generate_symbol_kines('000001', '30分钟', '20240101', '20240105')

# 转换为RawBar对象列表
bars = format_standard_kline(df, freq=Freq.F30)

# 创建CZSC分析对象
czsc_obj = CZSC(bars)
```

### 回测可视化

`czsc.utils.plotting.backtest` 模块已在二阶段清理 PR-C 删除。推荐做法：

- **权重回测报告**：`wbt.generate_backtest_report(dfw, ...)` 生成自包含 HTML（参见 `docs/examples/13_event_weight_backtest.py`）
- **缠论 + 多周期联立**：`czsc.utils.plotting.lightweight.plot_czsc{,_trader,_signals}` 输出 lightweight-charts HTML
- **单周期 K 线 + 缠论结构**：`czsc.utils.plotting.kline.KlineChart` / `plot_czsc_chart`
- **自定义统计图**：直接用 `plotly.express` / `plotly.graph_objects`，迁移示例见 `docs/migration/cleanup-non-czsc-core.md`

### 依赖管理（UV配置）

- 核心运行时依赖定义在 `pyproject.toml` 的 `[project.dependencies]` 中
- 日常开发工具在 `[dependency-groups].dev` 中，默认由 `uv sync` 安装，包含 pytest / Ruff / basedpyright / maturin 等
- `[project.optional-dependencies].dev` 是额外的 IPython / Jupyter 依赖；`test` 包含 pytest-cov 等，`all` 聚合可选依赖。两类 dev 配置不可混淆
- 修改 Python 依赖时同步 `uv.lock` 并重新同步环境；Rust 依赖由 Cargo workspace 管理，修改后检查 `Cargo.lock`
- 不为普通开发循环反复执行 `uv sync`；具体流程见「常用开发命令」

## 关键环境变量和设置

- `CZSC_VERBOSE` / `czsc_verbose`：是否打印详细日志（来自 `czsc.envs`）
- `CZSC_MIN_BI_LEN` / `czsc_min_bi_len`：最小笔长度，默认 6（来自 `czsc.envs`）
- `CZSC_MAX_BI_NUM` / `czsc_max_bi_num`：最大笔数量，默认 50（来自 `czsc.envs`）
- 大小写两种写法都接受，大写优先；构造器显式参数优先级最高
- 缓存目录自动管理，具备大小监控功能

## 缓存管理

项目大量使用磁盘缓存：
- 缓存位置：`czsc.home_path`（顶层）或 `czsc.utils.data.cache.home_path`（实际定义处）
- 清除缓存：`czsc.empty_cache_path()`
- 监控大小：`czsc.get_dir_size(czsc.home_path)`
- 当缓存超过1GB时 `czsc.welcome()` 会显示清理提示

## 可视化（Plotly + HTML）

可视化使用 Plotly 和 lightweight-charts，输出方式：

- `czsc.utils.plotting.kline.KlineChart` / `plot_czsc_chart`：单周期 K 线 + 缠论结构（plotly Figure，可 `fig.show()` 或写 HTML）
- `czsc.utils.plotting.weight.*`：权重时序图（plotly）
- `czsc.utils.plotting.lightweight.plot_czsc{,_trader,_signals}`：lightweight-charts 自包含 HTML，多周期联立 + 信号叠加

使用 `plot_czsc`、`plot_czsc_trader` 或 `plot_czsc_signals` 生成自包含 HTML 后，可直接在浏览器中打开。

## Rust/Python 混合架构

项目核心算法用 Rust 实现，通过 PyO3 暴露给 Python：
- **构建方式**：`maturin + Rust workspace`，扩展模块名 `czsc._native`
- **唯一架构**：Rust 是缠论核心算法的唯一实现；Python 端不再保留任何回退（spec §3.1 / §3.4）
- **API 暴露**：所有面向用户的 API 都通过 `czsc.xxx` 顶层命名空间暴露，禁止用户感知 `czsc._native`
- **Python/Rust 分工**：见本文件顶部「🏛️ 开发宪法 · 第一条」。该条款是硬约束，与此冲突的任何"局部例外"都不成立。
- **类型 stub**：`czsc/py.typed` 标记类型支持；扩展模块 stub 位于 `czsc/_native/__init__.pyi`，由 `pyo3-stub-gen` 自动维护，生成命令见「Rust 检查与类型 stub」
- **构建环境约束**：Python ≥ 3.10；PyO3 / numpy 的版本以根目录 `Cargo.toml [workspace.dependencies]` 为准，stub 生成器版本以 `crates/czsc-python/Cargo.toml` 为准。`build.rs` 校验解释器与 Python 动态版本配置；wheel 使用 `abi3-py310`
- **版本号锁死**（PR-5）：crates.io 与 PyPI 必须使用同一版本号。**唯一版本源**是 `Cargo.toml [workspace.package].version`；`pyproject.toml` 用 `dynamic = ["version"]`，由 maturin 在打 wheel 时从 Cargo workspace 注入。`crates/czsc-python/build.rs` 会在编译期校验 pyproject.toml 仍然走 dynamic 路径，禁止硬编码 `version = "..."`
- **发版流程**：更新 `Cargo.toml [workspace.package].version` 时，同步 `[workspace.dependencies]` 中内部 crate 的精确版本约束及相关锁文件；CHANGELOG 必须列出 breaking changes。Rust 与 Python 产物需发布相同版本，流程以 `.github/workflows/rust-publish.yml` 和 `python-publish.yml` 为准：tag push 触发 Rust dry-run，Rust 真发需 `workflow_dispatch` 设置 `do_publish=true`；crate 按依赖顺序发布，`czsc-python` 仅用于构建 wheel，不发布到 crates.io。发布前验证见 `docs/release_checklist.md`
- **rs-czsc 关系**：czsc 一次性 fork rs-czsc 的 Rust 实现进本仓库，**不再做季度同步**；`tests/parity/` 已删除，不恢复独立的 rs-czsc 比对目录。现有 `tests/unit/test_core_parity.py` 的固定基线快照仍需维护

## 数据连接器支持

项目集成多个数据源连接器（见 `czsc/connectors/`）：
- `tq_connector.py`: 天勤数据源
- `ts_connector.py`: Tushare 数据源
- `ccxt_connector.py`: 数字货币数据源
- `jq_connector.py`: 聚宽 JQData 数据源，需要 `jq` extra
- `local_data.py`: 投研数据本地缓存接口

## 回测和策略研究框架

### 策略开发基础（`czsc/strategies.py`）
- `CzscStrategyBase`: 策略开发的抽象基类
- `CzscJsonStrategy`: JSON 配置化的策略实现
- 策略要素：品种参数、K线周期、信号配置、持仓策略
- 支持策略序列化和反序列化
- 研究入口统一指向 `czsc.research.run_research / run_replay / run_optimize_batch`（Rust 后端）
- **回测路径优先级**：能使用 `Signal -> Event -> Position` 体系表达的策略，尽量使用 `Position/Event` 体系进行回测，并通过 `CzscStrategyBase.backtest / replay` 产出 `holds / pairs / signals`；仅在纯权重时序、组合调仓、探索性研究等不适合表达为 `Position/Event` 的场景，才优先使用 `WeightBacktest` 或自定义权重序列回测。

### 自动研究脚本（`scripts/auto-czsc-quant/`）

- 脚本级实验编排器，复用 CZSC 的 Position/Event 回测；运行入口、数据源和候选协议见该目录的 `README.md`
- 相关测试位于 `tests/test_auto_czsc_quant.py`；核心算法变更仍应落在 Rust 后端

### 探索性数据分析（顶层 `czsc.*`）

- `monotonicity`: Rust 实现的单调性分析
- `mark_cta_periods / mark_volatility`: 位于 `czsc/utils/` 的独立模块
- `czsc/eda.py` 仍保留上述函数的 re-export，作为历史导入路径的兼容入口；新代码优先使用顶层 API

> 其余历史函数 `weights_simple_ensemble` / `cal_trade_price` / `cal_yearly_days` / `turnover_rate` 已于 PR-A 二阶段清理中删除；迁移说明见 [`docs/migration/cleanup-non-czsc-core.md`](docs/migration/cleanup-non-czsc-core.md)。

## 重要文档和资源

- [项目文档](https://s0cqcxuy3p.feishu.cn/wiki/wikcn3gB1MKl3ClpLnboHM1QgKf)
- [信号函数编写规范](https://s0cqcxuy3p.feishu.cn/wiki/wikcnCFLLTNGbr2THqo7KtWfBkd)
- [API文档](https://czsc.readthedocs.io/en/latest/modules.html)
- [B站视频教程](https://space.bilibili.com/243682308/channel/series)

## 示例代码和用例

项目维护的示例代码集中在 `docs/examples/` 下（如 `08_weight_backtest.py`、`13_lightweight_charts_html.py`、`15_lightweight_signals_html.py` 等）；HTML 路径示例（13/15）完整保留。

## 项目特色和最佳实践

1. **混合架构设计**: Rust性能优化 + Python灵活性
2. **多级别联立分析**: 支持多时间周期综合决策
3. **系统化信号体系**: 信号→事件→交易的完整流程
4. **丰富的数据源**: 支持A股、期货、数字货币等多市场
5. **完善的测试框架**: 统一的模拟数据生成和测试规范
6. **可视化工具**: Plotly + lightweight-charts HTML 输出
7. **策略研究工具**: CTA框架、参数优化、回测分析一体化
8. **代码质量优化**: 遵循 DRY、KISS、SOLID 原则
   - 使用模块级常量消除魔法值
   - 提取辅助函数减少代码重复
   - 完善的类型提示（Type Hints）
   - 清晰的函数职责分离
   - 保持向后兼容性的 API 设计
