# APS 冷轧排程系统 — 代码架构分析

> 本文档基于 `src/aps_cp_sat` 模块源码，描述当前工程版本的完整架构设计。

---

## 1. 整体定位

`aps_cp_sat` 是一个面向冷轧生产线的**约束规划排程引擎**，基于 Google OR-Tools CP-SAT 和构造性大邻域搜索（ALNS）混合架构。系统接收订单集（Excel），经过分层处理后输出每条产线的排程计划（已排订单 + 剔除订单）。

当前工程模式仅允许 `constructive_lns_search` 系列 profile，所有 joint_master 路径已被 Profile Guard 硬拦截。

---

## 2. 分层架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│  ColdRollingPipeline.run()   ← 唯一公开入口（冷轧主管道）         │
└────────────┬────────────────────────────────────────────────────┘
             │ ColdRollingRequest
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0: PREPROCESS（数据预处理）                               │
│  preprocess/pipeline.py                                         │
│  io/readers.py → load_orders / load_grade_catalog               │
│  输出: orders_df                                                │
└────────────┬────────────────────────────────────────────────────┘
             │ orders_df
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: TRANSITION（过渡模板构建）                             │
│  transition/template_builder.py                                 │
│  transition/template_pruning.py                                 │
│  transition/bridge_rules.py                                      │
│  输出: transition_pack = { templates, summaries, prune_summaries, │
│                             build_debug, candidate_graph }       │
└────────────┬────────────────────────────────────────────────────┘
             │ orders_df + transition_pack
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: MODEL（求解引擎）                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ solve_master_model()  ← 统一分发层                           │  │
│  │ Profile Guard → constructive_lns branch (only allowed)      │  │
│  └────────────┬───────────────────────────────────────────────┘  │
│               ▼                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ run_constructive_lns_master()  ← ALNS 主循环               │  │
│  │                                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Layer 2a: build_constructive_sequences()              │  │  │
│  │  │ constructive_sequence_builder.py                     │  │  │
│  │  │ 贪心链式构建：沿模板边走，生成 ConstructiveChain 列表  │  │  │
│  │  └────────────┬─────────────────────────────────────────┘  │  │
│  │               ▼                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Layer 2b: cut_sequences_into_campaigns()             │  │  │
│  │  │ campaign_cutter.py                                   │  │  │
│  │  │ 将长链切段：吨位窗口边界 → CampaignSegment            │  │  │
│  │  │ underfilled → underfilled_segments（不丢弃）          │  │  │
│  │  └────────────┬─────────────────────────────────────────┘  │  │
│  │               ▼                                            │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Layer 2c: solve_local_insertion_subproblem()        │  │  │
│  │  │ local_inserter_cp_sat.py                            │  │  │
│  │  │ CP-SAT 局部重插入：ALNS destroy/repair 子问题       │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────┬───────────────────────────────────────────────┘  │
│               ▼                                                  │
│  输出: planned_df / dropped_df / rounds_df / engine_meta        │
└────────────┬────────────────────────────────────────────────────┘
             │ ColdRollingResult
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: DECODE（结果解码）                                      │
│  decode/result_decoder.py                                       │
│  decode/joint_solution_decoder.py                               │
│  输出: ColdRollingResult（带 decode 注解）                      │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: VALIDATE（结果验证）                                    │
│  validate/solution_validator.py                                  │
│  validate/checks.py                                             │
│  validate_model_equivalence() / validate_solution_summary()     │
└────────────┬────────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: EXPORT（结果导出）                                      │
│  io/result_writer.py                                            │
│  输出: output_path/*.xlsx                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块职责

### 3.1 `cold_rolling_pipeline.py` — 主管道门面

```python
class ColdRollingPipeline:
    def run(self, req: ColdRollingRequest) -> ColdRollingResult:
        # 1. Profile Guard（强制 constructive_lns_search）
        req = self._enforce_constructive_lns_profile(req)
        # 2. 数据加载
        orders_df = prepare_orders_for_model(req.orders_path, ...)
        # 3. 模板构建 + candidate_graph
        transition_pack = build_transition_templates(orders_df, req.config)
        transition_pack = self._attach_candidate_graph(orders_df, transition_pack, req.config)
        # 4. 求解
        schedule_df, rounds_df, dropped_df, engine_meta = solve_master_model(req, ...)
        # 5. 解码
        result = decode_solution(result)
        # 6. 验证
        summary = validate_solution_summary(result, result.config.rule)
        eq = validate_model_equivalence(result.schedule_df, ...)
        # 7. 导出
        export_schedule_results(result, ...)
```

关键职责：
- **Profile Guard**：仅允许 `constructive_lns_*` 系列 profile；空/默认 profile 自动纠正为 `constructive_lns_search`
- 统一日志前缀 `[APS][*]`
- 统一 engine_meta 字段标准化（`UNIFIED_ENGINE_META_FIELDS`）
- 诊断数据收集与报告

### 3.2 `preprocess/` — 数据准备

| 文件 | 职责 |
|------|------|
| `io/readers.py` | 读取 Excel（订单、钢种目录）；字段归一化（line_capability 等） |
| `preprocess/pipeline.py` | `prepare_orders_for_model()` — 数据加载入口 |

### 3.3 `transition/` — 过渡模板系统

| 文件 | 职责 |
|------|------|
| `template_builder.py` | 从 orders_df 构建候选模板对（from_order_id, to_order_id）；按 line 分组 |
| `bridge_rules.py` | 桥接边合法性规则（PC组判断、温度重叠、厚度约束、txt 等） |
| `template_pruning.py` | 按度数/topK 剪枝；保留有效模板对 |
| `template_cost.py` | 模板边成本计算 |
| `bridge_expansion_contract.py` | 桥接展开契约（当前 disabled） |

**输出 `transition_pack`**：
```python
{
    "templates": DataFrame,        # 有效模板对（from_order_id, to_order_id, line, bridge_count）
    "summaries": list[TemplateSummary],  # 每条线的覆盖率统计
    "prune_summaries": list[PruneSummary],  # 剪枝统计
    "build_debug": list[dict],     # 构建耗时明细
    "candidate_graph": CandidateGraph,  # 候选图对象（见 3.4）
    "candidate_graph_diagnostics": dict,
}
```

### 3.4 `model/candidate_graph.py` — 候选图（ALNS 搜索空间）

```python
build_candidate_graph(orders_df, tpl_df, cfg) → CandidateGraph
```

构建边类型：
- **DIRECT_EDGE**：直接相邻，有效模板对
- **REAL_BRIDGE_EDGE**：实物桥接（PC 组内，经 bridge_rules 验证）
- **VIRTUAL_BRIDGE_FAMILY_EDGE**：虚拟家族桥接（跨 PC 组，允许的虚拟宽度/厚度级别）

过滤器（逐层递减）：
```
width 过滤 → thickness 过滤 → temp 过滤 → group 过滤 → max_virtual_chain 过滤
```

diagnostics 字段追踪每层过滤数量。

### 3.5 `model/constructive_sequence_builder.py` — Layer 2a 构造层

```python
build_constructive_sequences(orders_df, transition_pack, cfg) → ConstructiveBuildResult
```

**策略**：贪心最长路径沿边扩展
1. 从入度=0 或最高 score 的订单开始
2. 沿 `candidate_graph` 走，优先选择低 bridge_count 边
3. 同 line 订单链独立构建
4. 死岛订单（无任何边）→ `dropped_seed_orders`

**输出**：
```python
@dataclass
class ConstructiveChain:
    chain_id: str
    line: str                # big_roll / small_roll
    order_ids: List[str]     # 顺序（继承自模板）
    total_tons: float
    edge_count: int
    start_order_id: str
    end_order_id: str
    steel_groups: List[str]
```

### 3.6 `model/campaign_cutter.py` — Layer 2b 切段层

```python
cut_sequences_into_campaigns(chains_by_line, orders_df, cfg, tpl_df) → CampaignCutResult
```

**切段策略（四策略窗口修复）**：

对每条线的每条链，以 `prev_seg + tail_seg` 构成窗口，依次尝试：

| 策略 | 触发条件 | 修复动作 |
|------|---------|---------|
| `RECUT_TWO_SEGMENTS` | tail.underfilled | 重切 prev+tail 窗口，找最优切分位置 |
| `SHIFT_FROM_PREV` | shift_gap ≤ 阈值 | 从 prev 尾部转移订单到 tail |
| `TAIL_FILL_FROM_DROPPED` | gap ≤ 阈值 | 从已剔除订单池中选候选补入 tail |
| `MERGE_WITH_PREV` | 最后兜底 | 将 tail 合并入 prev（若 multiplicity 允许） |

**切段原因（CutReason）**：
- `MAX_LIMIT`：加下一单会超 ton_max
- `TARGET_REACHED`：在窗口内，加下一单会加大偏离
- `NO_SUCCESSOR`：链已耗尽
- `TAIL_UNDERFILLED`：尾段 < ton_min

**underfilled_segments 处理**：
- 不丢弃，放入单独列表
- 由 `_reconstruct_underfilled_segments()` 尝试用 bridge 边重建
- 重建失败仍 underfilled → 计入 dropped

**关键数据结构**：
```python
@dataclass
class CampaignSegment:
    line: str
    campaign_local_id: int   # 段在线内的局部 ID（1-based）
    order_ids: List[str]
    total_tons: float
    cut_reason: CutReason
    is_valid: bool           # total_tons >= ton_min
```

### 3.7 `model/local_inserter_cp_sat.py` — Layer 2c 局部插入

```python
solve_local_insertion_subproblem(req: LocalInsertRequest) → LocalInsertResult
```

**职责**：ALNS destroy/repair 循环中的 repair 子问题
- 输入：fixed_order_ids（必须保留顺序）+ candidate_insert_ids（可插入/拒绝）
- 建模：CP-SAT AddCircuit（哈密顿路）
- 约束：严格模板边（strict_template_edges=True）
- 输出：`InsertStatus`（OPTIMAL/FEASIBLE/INFEASIBLE/NO_IMPROVEMENT）

### 3.8 `model/constructive_lns_master.py` — ALNS 主循环

```python
run_constructive_lns_master(orders_df, transition_pack, cfg, random_seed) → ConstructiveLnsResult
```

**ALNS 流程**：

```
初始解构建 (Layer 2a + 2b)
    ↓
初始 planned_df / dropped_df
    ↓
┌──────────────────────────────────┐
│  for round in range(n_rounds):   │
│  ┌────────────────────────────┐ │
│  │ 1. DESTRUCTION             │ │
│  │    选 neighborhood         │ │
│  │    LOW_FILL_SEGMENT        │ │
│  │    HIGH_DROP_PRESSURE      │ │
│  │    HIGH_VIRTUAL_USAGE      │ │
│  │    TAIL_REBALANCE          │ │
│  │    SMALL_ROLL_RESCUE       │ │
│  │    → 移除部分订单          │ │
│  └────────────┬───────────────┘ │
│               ▼                  │
│  ┌────────────────────────────┐ │
│  │ 2. REPAIR (Layer 2c)       │ │
│  │    CP-SAT 局部插入          │ │
│  │    尝试将被删订单重新插入   │ │
│  └────────────┬───────────────┘ │
│               ▼                  │
│  ┌────────────────────────────┐ │
│  │ 3. ACCEPTANCE              │ │
│  │    多准则比较               │ │
│  │    accepted → 更新 best     │ │
│  │    rejected → rollback     │ │
│  └────────────┬───────────────┘ │
│               ▼                  │
│  早停检查 (no_improve >= N)      │
└──────────────────────────────────┘
    ↓
输出 planned_df / dropped_df / rounds_df / diagnostics
```

**DropReason 枚举**（标注每个被剔除订单的原因）：
- `DEAD_ISLAND_ORDER`：无模板边
- `TAIL_UNDERFILLED`：段吨位不足
- `LNS_REPAIR_REJECTED`：CP-SAT 拒绝
- `NO_FEASIBLE_LINE`：无兼容产线
- `CONSTRUCTIVE_REJECTED`：构造层拒绝
- `FINAL_SEGMENT_TEMPLATE_MISS`：段内模板对缺失

### 3.9 `decode/result_decoder.py` — 解码层

```python
decode_solution(result: ColdRollingResult) → ColdRollingResult
```

职责：
- 将 engine 内部表示（campaign_id_hint, order_ids 顺序）解码为最终 DataFrame 列
- 处理 `candidate_graph` 展开（VIRTUAL_BRIDGE_EDGE → 具体虚拟订单）
- 标注 `selected_edge_type`（DIRECT/REAL_BRIDGE/VIRTUAL_BRIDGE）
- 记录 decode 顺序完整性检查（decode_order_integrity_ok）

### 3.10 `validate/solution_validator.py` — 验证层

```python
validate_solution_summary(result, rule) → dict
validate_model_equivalence(schedule_df, tpl_df) → dict
```

检查项：
- `template_pair_ok`：所有相邻对均在模板中
- `adjacency_rule_ok`：无宽度跳变、厚度冲突、温度重叠
- `bridge_expand_ok`：桥接展开路径存在
- campaign_ton_min/max 硬约束

### 3.11 `io/result_writer.py` — 导出层

```python
export_schedule_results(result, output_path, ...)
```

输出：
- `schedule_*.xlsx`：已排订单（含 edge_type 中文标注）
- `dropped_*.xlsx`：剔除订单（含 drop_reason 层级分类）
- 中文标签映射（DIRECT_EDGE→直接相邻，REAL_BRIDGE→实物桥接，VIRTUAL_BRIDGE→虚拟桥接）

---

## 4. 配置驱动机制

### 4.1 `config/parameters.py` — Profile 工厂

```python
build_profile_config(profile_name, validation_mode, production_compatibility_mode) → PlannerConfig
```

支持的 profile：
| Profile | 特性 |
|---------|------|
| `constructive_lns_search` | 生产主力；ALNS 60轮，早停3轮，max_total=10 |
| `constructive_lns_debug_acceptance` | 放宽 partial acceptance 阈值，仅用于调试 |
| `constructive_lns_direct_only_baseline` | 禁用所有桥接边（direct_only），基准测试用 |
| `constructive_lns_real_bridge_frontload` | 启用 REAL_BRIDGE_EDGE，禁用虚拟桥接 |

### 4.2 `config/model_config.py` — 模型参数

关键字段（按功能分组）：

**ALNS 控制**：
```python
constructive_lns_rounds: int = 60
lns_early_stop_no_improve_rounds: int = 3
lns_max_total_rounds: int = 10
lns_min_rounds_before_early_stop: int = 4
```

**Edge 策略**：
```python
allow_virtual_bridge_edge_in_constructive: bool = False
allow_real_bridge_edge_in_constructive: bool = False
bridge_expansion_mode: str = "disabled"
repair_only_real_bridge_enabled: bool = True
repair_only_virtual_bridge_enabled: bool = False
repair_only_virtual_bridge_pilot_enabled: bool = False
```

**Tail Fill 增强**（2-pass fill）：
```python
tail_fill_max_inserts_per_tail: int = 2       # 最多2次连续补吨
tail_fill_second_pass_gap_limit: float = 30.0 # 第2次 pass 触发阈值
```

**模板剪枝**：
```python
global_prune_max_pairs_per_from: int = 0
template_top_k: int = 40
max_virtual_chain: int = 5
```

### 4.3 `config/rule_config.py` — 硬约束规则

```python
@dataclass
class RuleConfig:
    campaign_ton_min: float = 500.0    # 最小吨位
    campaign_ton_max: float = 2000.0   # 最大吨位
    campaign_ton_target: float = 1500.0 # 目标吨位
    # 过渡规则
    reverse_width_limit: int = 2
    reverse_width_total: int = 8
    # 虚拟边规则
    virtual_width_levels: List[float] = [1250, 1500, 1800]
    virtual_thickness_levels: List[float] = [2.0, 2.5, 3.0]
    virtual_temp_min: float = 580.0
    virtual_temp_max: float = 700.0
```

### 4.4 `config/score_config.py` — 软约束评分

```python
@dataclass
class ScoreConfig:
    unassigned_real: int = 1000         # 未排订单惩罚
    slot_isolation_risk_penalty: int = 80
    slot_pair_gap_risk_penalty: int = 60
    slot_span_risk_penalty: int = 40
    slot_order_count_penalty: int = 60
```

---

## 5. 数据流与关键接口

### 5.1 顶层数据类

```python
@dataclass(frozen=True)
class ColdRollingRequest:
    orders_path: Path
    steel_info_path: Path
    output_path: Path
    config: PlannerConfig

@dataclass(frozen=True)
class ColdRollingResult:
    schedule_df: pd.DataFrame    # 已排订单
    rounds_df: pd.DataFrame     # 每轮 ALNS 统计
    output_path: Path
    dropped_df: pd.DataFrame | None  # 剔除订单
    engine_meta: dict | None
    config: PlannerConfig | None
```

### 5.2 内部数据类

| 类 | 定义文件 | 用途 |
|----|---------|------|
| `ConstructiveChain` | constructive_sequence_builder.py | Layer 1 输出：长订单链 |
| `CampaignSegment` | campaign_cutter.py | Layer 2 输出：候选段 |
| `ConstructiveBuildResult` | constructive_sequence_builder.py | Layer 1 总输出 |
| `CampaignCutResult` | campaign_cutter.py | Layer 2 总输出 |
| `LocalInsertRequest` | local_inserter_cp_sat.py | Layer 3 输入 |
| `LocalInsertResult` | local_inserter_cp_sat.py | Layer 3 输出 |
| `ConstructiveLnsResult` | constructive_lns_master.py | ALNS 最终输出 |

### 5.3 关键函数调用链

```
ColdRollingPipeline.run()
  └→ prepare_orders_for_model()
  └→ build_transition_templates()
  └→ build_candidate_graph()
  └→ solve_master_model()
       └→ run_constructive_lns_master()
            ├→ build_constructive_sequences()          [Layer 2a]
            ├→ cut_sequences_into_campaigns()           [Layer 2b]
            │    ├→ _reconstruct_underfilled_segments()
            │    └→ _validate_segment_template_pairs()
            ├→ _segments_to_planned_df()               [初始化 planned_df]
            └→ ALNS loop:
                 ├→ _select_neighborhood()               [DESTRUCTION]
                 ├→ _remove_orders_from_segments()       [DESTRUCTION]
                 ├→ solve_local_insertion_subproblem()  [REPAIR - Layer 2c]
                 └→ _accept_or_reject()                  [ACCEPTANCE]
  └→ decode_solution()
  └→ validate_model_equivalence() / validate_solution_summary()
  └→ export_schedule_results()
```

---

## 6. 候选图架构（Candidate Graph）

`candidate_graph.py` 构建了一个有向图 `CandidateGraph`：

```python
@dataclass
class CandidateGraph:
    nodes: List[str]                    # 所有 order_id
    edges: List[CandidateEdge]          # 所有候选边
    by_line: Dict[str, List[CandidateEdge]]  # 按 line 索引
    diagnostics: Dict                   # 每层过滤计数
```

**边类型**：
```python
DIRECT_EDGE            = "direct"
REAL_BRIDGE_EDGE       = "real_bridge"
VIRTUAL_BRIDGE_FAMILY_EDGE = "virtual_bridge_family"
```

**构建流水线**（`build_candidate_graph`）：
```
orders_df × orders_df → candidate pairs
    ↓ width 过滤
    ↓ thickness 过滤
    ↓ temp 过滤（PC组内）
    ↓ group 过滤（跨PC需bridge_rules验证）
    ↓ max_virtual_chain 过滤
→ CandidateGraph
```

---

## 7. Campaign Cutter 窗口修复控制流

```
对每条 line 的每个 (prev_seg, tail_seg) 窗口：

  ┌─────────────────────────────────────────────────┐
  │ 1. RECUT_TWO_SEGMENTS（首选）                  │
  │    重切窗口，找最优位置，评分：                 │
  │    score = (left_gap, right_gap, right_order_cnt)│
  │    → FILL_FULL_VALID / FILL_PARTIAL_PROGRESS /  │
  │      FILL_NO_PROGRESS / FILL_FAILED             │
  │    FILL_PARTIAL/PROGRESS → repaired=True         │
  └────────────────────┬────────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────────┐
  │ 2. SHIFT_FROM_PREV（次选，gap ≤ shift_gap_limit）│
  │    从 prev 尾部转移订单到 tail                  │
  └────────────────────┬────────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────────┐
  │ 3. TAIL_FILL_FROM_DROPPED（第三，gap ≤ fill_gap）│
  │    候选池多阶段过滤：                           │
  │    placed_oids → 同line → template_pair → gap  │
  │    → 2-pass fill (max_inserts_per_tail=2)       │
  │    第2次 pass 触发条件：gap ≤ second_pass_limit │
  │    FILL_PARTIAL/PROGRESS → repaired=True        │
  └────────────────────┬────────────────────────────┘
                       ↓
  ┌─────────────────────────────────────────────────┐
  │ 4. MERGE_WITH_PREV（最终兜底）                  │
  │    断言：not repaired（防止 FILL 后继续 MERGE） │
  │    multiplicity 检查通过 → 合并                │
  │    multiplicity 失败 → window 跳过             │
  └─────────────────────────────────────────────────┘
```

**Fill 结果枚举**：
```python
FILL_FULL_VALID        # tail 达到 ton_min
FILL_PARTIAL_PROGRESS  # 有进展但 tail 仍 underfilled
FILL_NO_PROGRESS       # 无有效候选
FILL_FAILED            # 异常
```

---

## 8. Domain 模型

```python
# src/aps_cp_sat/domain/models.py
@dataclass(frozen=True)
class ColdRollingRequest:
    orders_path: Path
    steel_info_path: Path
    output_path: Path
    config: PlannerConfig

@dataclass(frozen=True)
class ColdRollingResult:
    schedule_df: pd.DataFrame
    rounds_df: pd.DataFrame
    output_path: Path
    dropped_df: pd.DataFrame | None
    engine_meta: dict | None
    config: PlannerConfig | None
```

---

## 9. 约束层级（Rule Semantics）

| 约束 | 级别 | 说明 |
|------|------|------|
| line_compatibility | HARD | 订单必须分配到兼容产线 |
| adjacent_transition_feasibility | HARD | 相邻订单必须有模板边 |
| campaign_ton_upper_bound | HARD | ≤ ton_max |
| campaign_ton_lower_bound | HARD | ≥ ton_min（v2 升为 HARD） |
| unassigned_real_orders | STRONG_SOFT | 未排订单惩罚 |
| virtual_slab_ratio/quantity | STRONG_SOFT | 虚拟边使用量惩罚 |
| reverse_width_count/total | HARD | 宽度反转次数限制（在模板规则中） |

---

## 10. 入口点与运行示例

```python
# examples/run_cold_rolling_pipeline.py
from aps_cp_sat.cold_rolling_pipeline import ColdRollingPipeline
from aps_cp_sat.domain.models import ColdRollingRequest
from aps_cp_sat.config.parameters import build_profile_config

req = ColdRollingRequest(
    orders_path=Path("data_orders.xlsx"),
    steel_info_path=Path("data_steel_info.xlsx"),
    output_path=Path("output.xlsx"),
    config=build_profile_config("constructive_lns_search"),
)
result = ColdRollingPipeline().run(req)
```

---

## 11. 目录结构

```
src/aps_cp_sat/
├── cold_rolling_pipeline.py   # 主管道（入口）
├── cold_rolling_scheduler.py  # 兼容层（legacy）
├── scheduler.py               # 简单调度（独立 toy）
├── constraints_config.py      # 约束配置
│
├── api/                       # API 层
├── compat/                    # 兼容层
├── config/                    # 配置（ModelConfig, RuleConfig, ScoreConfig, parameters）
├── decode/                    # 解码（result_decoder, joint_solution_decoder）
├── domain/                   # 领域模型（ColdRollingRequest/Result）
├── io/                        # IO（readers, result_writer）
├── model/                     # 求解模型
│   ├── master.py              # solve_master_model（分发层 + Profile Guard）
│   ├── constructive_lns_master.py  # ALNS 主循环
│   ├── constructive_sequence_builder.py  # Layer 2a
│   ├── campaign_cutter.py     # Layer 2b（四策略窗口修复）
│   ├── local_inserter_cp_sat.py  # Layer 2c
│   ├── candidate_graph.py     # 候选图
│   ├── candidate_graph_types.py  # 边类型常量
│   ├── fallback_policy.py
│   ├── feasibility_evidence.py
│   └── ...
├── persistence/              # 持久化（service.py, *.sql）
├── preprocess/                # 预处理（pipeline, grade_catalog, order_preparation）
├── repair/                   # 修复层（当前 no-op）
├── rules/                    # 规则注册（RULE_REGISTRY, RuleKey）
├── transition/               # 过渡模板（builder, pruning, bridge_rules, cost）
└── validate/                 # 验证层（solution_validator, checks）
```

---

## 12. 近期关键变更（2026-04）

| 变更 | 文件 | 说明 |
|------|------|------|
| 2-pass tail fill | campaign_cutter.py, parameters.py | `tail_fill_max_inserts_per_tail=2`，gap≤30触发第2次 |
| FILL后repaired标志 | campaign_cutter.py | FILL partial/full后立即repaired=True，阻止MERGE继续 |
| placed_oids补全 | campaign_cutter.py | 扩展为`segments + underfilled_segments`，防止从underfilled偷单 |
| TAIL_FILL_POOL日志 | campaign_cutter.py | 多阶段候选过滤计数日志 |
| Profile Guard强化 | cold_rolling_pipeline.py, master.py | 仅允许constructive_lns_* profile |
| DROP_PRIORITY统一 | master.py | 全局统一剔除优先级：GLOBAL_ISO > TON_WINDOW > NO_FEASIBLE > BRIDGE > ADJACENCY > ISO > LOW |
| bridge edge policy | model_config.py | allow_virtual/allow_real/bridge_expansion_mode 三元策略 |
