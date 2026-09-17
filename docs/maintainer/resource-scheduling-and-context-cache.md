# NInfer 资源调度与上下文缓存

本文定义 NInfer 在有限 Device/Host 资源下选择、验证并提交上下文状态的规则。它是 admission
materialization、prefix reuse、cache retention 和 pressure planning 的维护者权威。

本文解决的问题是：

> Scheduler 已经选择一个等待请求后，应从哪个可恢复位置开始，保留或降级哪些 inactive context，
> 才能以最低预测代价得到一个物理可执行且不会破坏 active request 完成保证的终态？

顶层请求顺序和生命周期由 [Engine 架构](engine-architecture.md)定义。KV page、address space、
replica 和 consumer view 的物理合同由 [Paged KV Context Store](paged-kv-cache.md)定义。

---

## 1. 术语与边界

本文使用以下术语：

| 术语 | 含义 |
|---|---|
| owner | 一个 active sequence、private continuation 或 shared prefix |
| checkpoint | owner 内某个可恢复 frontier 的不可变完整状态 |
| candidate | incoming request 可以采用的一个 root 或 exact checkpoint source |
| planning target | candidate 与全部受影响 inactive owners 的完整逻辑终态 |
| reservation | 已从全局可用容量中扣除、为未来物化或事务目的保留的容量 |
| resource plan | Program 针对一个 target seal 的有序物理 transition |
| placement | 一个完整 StateImage 或一个 KV logical page 的 Device/Host replica 状态 |

本文后续简称 planning target 为 `target`，表示一次资源决策的终态。

Prefix reuse 是 admission 的一种来源选择。它复用具有完整身份和状态覆盖的已计算前缀，减少重复
prefill，保持模型语义和请求顺序。不同 prefill 分块或数值路径之间不要求 logits 或生成 token 完全相同。

当前产品条件为单 GPU、单 resident model、固定 `max_concurrency=1..8` 和非抢占 active requests。
由此得到两个基本规则：

1. Scheduler 先确定本次尝试的 request，资源层只优化该 request 的 materialization；
2. 一个 request 发布为 Active 后，其最大合法执行范围已经获得完整资源保证，inactive cache policy
   不能再借用这部分容量。

---

## 2. 决策所有权

### 2.1 Scheduler

Scheduler 选择：

- 当前 FIFO head 或已经证明安全的 backfill request；
- admission 尝试时机；
- prefill、control 和 decode 的执行顺序。

Scheduler 不读取 checkpoint placement、victim、page 或 allocator facts。

### 2.2 ResourceManager

ResourceManager 拥有逻辑缓存策略：

- private/shared catalog 与 SessionIndex；
- candidate shortlist 和 logical claim；
- retention class、命中历史与 publication order；
- 跨 candidates 的 target 搜索和成本比较；
- Program 终态结果的逻辑采用。

ResourceManager 可以知道“某个 checkpoint 可从 Device、Host 或 root 恢复”的 Program summary，
但不保存物理 occupancy、refcount、free page 或 Host arena 几何的副本。

ResourceManager 对 active request 的逻辑保护保存为 owner edge，而不是 catalog capability 或引用计数镜像。
Capability generation 是 planning/transaction 的瞬时结构快照；owner edge 是持续到 active request terminal
settlement 的逻辑 lease。另一个 reader 改变同一 owner 的 Device/Host residency 时可以推进 generation，但不
使已有 owner edge stale。所有 active logical reference counts 都从这些 edges 派生。

### 2.3 Program

Program 是唯一物理 authority，拥有：

- State 与 KV 的真实 objects、references 和 replicas；
- Device pools、Host stores、allocators 和 reservations；
- continuation completeness 与 exact prefix identity；
- Move、Fork、COW、transfer 和 last-reference release；
- 完整 post-state、每个 transition stage 的物理可行性；
- 与当前资源状态绑定的 `ResourcePlan`。

调用关系为：

```text
Scheduler selects one request
        │
        ▼
ResourceManager enumerates candidates and compares targets
        │
        ▼
Program projects each complete target from real physical state
        │
        ▼
Program seals and executes one ResourcePlan
        │
        ▼
ResourceManager adopts one stable ResourceResult
```

逻辑策略与物理事实通过 candidate assessment、opaque target handle、`ResourcePlan` 和
`ResourceResult` 交互，不通过一份跨层共享的物理账本交互。

---

## 3. 资源模型

### 3.1 容量轴

Program 在启动时建立一个固定容量向量：

```text
PhysicalCapacity = {
    Device StateImage slots,
    Host StateImage slots,
    Device page groups for each typed KV pool,
    Host KV arena bytes and allocator geometry,
}
```

容量依据加载后的模型配置、实际权重绑定、启用组件及启动选项推导。权重的物理表示影响执行
workspace 和可用预算；KV/GDN 的数学状态、具体存储和容量由模型实现与 Program 分别负责。

这些轴相互独立。一个资源轴的余量不能补偿另一个轴的缺口。Host KV 的总空闲字节也不能替代
allocator 对具体 extent 几何的可分配性判断。

Engine 另有固定的逻辑容量：

- active lanes；
- private continuation catalog slots；
- shared prefix catalog slots；
- 每个 continuation 的 checkpoint/anchor 上限。

Lane 和 catalog slot 不计入物理 bytes，但 target 只有同时满足逻辑容量和物理容量才可发布。

### 3.2 唯一物理占用

对任一物理资源维度 \(r\)，状态 \(S\) 的占用为：

\[
Used_r(S)=
\sum_{x\in UniqueAllocations(S)} size_r(x)
+
\sum_{y\in ConcreteReservations(S)} size_r(y)
\]

`UniqueAllocations` 包括：

- Device/Host StateImage replicas；
- Device KV page replicas；
- Host KV extents。

`ConcreteReservations` 包括：

- active request 尚未 materialize 的未来增长；
- partial-tail COW destination；
- resource transition 的 copy destination 和 replacement capacity。

同一个物理 allocation 被多个 checkpoints 或 address spaces 引用时只计一次。已经物化的容量计
allocation，尚未物化但被保护的容量计 reservation；同一容量单位不能同时计入两项。

全局物理占用与 owner-exclusive resources 是两个不同视图。前者直接来自 State/KV stores 的 unique
allocations 与 concrete reservations；后者只表示单独移除一个 owner 时能够结算的资源。一个同时被
private checkpoint 与 shared prefix 引用的 allocation 在全局占用中计一次，但在任一 owner 的 exclusive
resources 中都不计。Owner-exclusive resources 可用于 eviction effect、ownership transfer 和 active
entitlement 校验，不能用于判断一个 checkpoint 是否具有 resident replica；后者只能查询对应 physical
store。

每个稳定状态和 transition 中间状态都必须满足：

\[
Used_r(S)\le Capacity_r
\]

### 3.3 完整 post-state 与联合回收

资源回收由最终剩余引用决定。对一组 logical actions \(V\)：

\[
Reclaim_r(S,V)=
Used_r(S)-Used_r(PostState(S,V))
\]

`PostState` 必须同时应用：

- selected source 的 retain 或 consume；
- 所有 victim owner/checkpoint outcomes；
- 所有 surviving Device/Host placements；
- active destination references 和 reservations；
- 零引用 objects 的 release。

因此通常有：

\[
Reclaim(S,A\cup B)\ne Reclaim(S,A)+Reclaim(S,B)
\]

删除一个 alias 可能不释放任何 allocation；两个 owners 一起删除最后一批 references 时，同一 allocation
只释放一次。ResourceManager 不计算或相加 per-owner reclaim credit。

Move、Fork、partial-tail COW 和 source placement 也由完整 post-state 决定。一个 source 在初始状态中
存在多个引用，并不意味着最终一定 Fork；反之，保留任一 checkpoint 所需的引用后也不能把它当作 Move。

### 3.4 有序阶段峰值

一个物理 plan 是稳定状态之间的有序阶段：

\[
S_0\rightarrow S_1\rightarrow\cdots\rightarrow S_n
\]

它可执行当且仅当：

\[
\forall r,\qquad
\max_{0\le i\le n} Used_r(S_i)\le Capacity_r
\]

阶段顺序必须表达真实 dependency：

```text
reserve destination
  -> copy
  -> verify and publish replacement
  -> release old replica
```

State restore、KV H2D/D2H、D2D fork、partial-tail COW 和 Host extent replacement 都遵循这一规则。
最终净占用不能掩盖 copy-before-release 的瞬时峰值。

### 3.5 Resource revision

Program 暴露单调 `resource_revision`，使只读 assessment 和 sealed plan 与一个稳定物理状态绑定。
以下变化会推进 revision：

- owner/reference topology 改变；
- inactive Device/Host placement 改变；
- global free/reserved capacity 改变；
- 会改变后续 allocator feasibility 的几何状态改变。

Active request 在自己的 reservation 内进行 mapped/reserved 转换、推进 committed frontier 或更新
state 内容时，不改变全局可用容量，因此不推进 revision。

一个 open resource transition 的中间阶段不向外发布 revision。它到达稳定 commit/abort 状态后，
若全局物理事实发生变化，只推进一次。Handle 的 owner/generation validation 独立阻止 slot reuse
产生的 stale capability。

---

## 4. Continuation 与 checkpoint

### 4.1 完整恢复条件

当前 Qwen3.5 模型同时包含可分页 Full Attention KV 和不能从任意较晚状态无损回退的 recurrent state。
因此某个 frontier 可以复用，当且仅当 Program 能证明：

1. 存在该 frontier 的完整 StateImage；
2. Main KV 满足 target 定义的 typed coverage；
3. selected backend 的 KV 与 fixed state 满足同一 continuation；
4. token、position、Vision 和 mode identity 与 incoming prompt 精确一致。

只有 token match、KV bytes 或 page match 时，缺少的是完整 continuation，不是一次部分 cache hit。

### 4.2 Continuation 组织

Private continuation 表示一条可继续演化的会话历史：

```text
PrivateContinuation
├── exact target/token history
├── Main KV address space
├── optional backend KV address space
└── one or more immutable checkpoints
```

Shared prefix 是不可变的复用来源，可以被多个 private branches Fork。一个 continuation 内的多个
checkpoints 可以引用相同 address space 的不同 prefix，但每个 checkpoint 都有自己的完整 StateImage
identity。

### 4.3 Checkpoint 种类

| Kind | 语义 |
|---|---|
| `SessionEndpoint` | 已完成请求的最新可继续状态 |
| `TurnClosure` | 当前 turn 中可替换 assistant suffix 之前的稳定状态 |
| `ResponseReplay` | generation opener 之前、可重新生成 response 的稳定状态 |
| `LongAnchor` | retention policy 选择的较早长上下文恢复点 |
| `SharedStablePrefix` | 多条历史共同使用的不可变稳定前缀 |

`TurnClosure` 和 `ResponseReplay` 的 frontier 必须位于下一请求可能替换的 assistant suffix 之前。
这样对话系统追加新的 user message 或改写生成尾部时，仍能复用稳定历史，而不会把可变 suffix 错当成
prefix identity 的一部分。

Checkpoint 发布后：

- identity、frontier 和 required coverage 不变；
- StateImage 与受保护的 KV prefix 不再原地修改；
- Device/Host placement 可以由 resource transition 改变。

### 4.4 Valid 与 Device-ready

Checkpoint \(c\) 有效，当且仅当：

- StateImage 至少有一个完整 Device 或 Host replica；
- 每个 required KV logical page 至少有一个 content epoch 一致且 coverage 足够的 Device 或 Host replica。

\[
\forall p\in RequiredPages(c),\qquad ValidDevice(p,c)\lor ValidHost(p,c)
\]

有效 checkpoint 可能需要 H2D restore。只有全部 execution requirements 已在 Device，且 active future
reservation 可以取得时，它才是 Device-ready。

### 4.5 Exact identity

Program 的 exact verification 可以使用：

- token IDs 和 token types；
- position/MRoPE axes 与 RoPE delta；
- Vision spans 和 media digest；
- template/runtime mode；
- checkpoint frontier。

Session key、marker、hash 和 prefix index 只缩小 candidate 集合，不证明命中。

`rewrite_execution_frontiers` 也是 exact identity 的一部分。它记录 replay/root prefill 必须分段的 exact
token frontiers，使重建路径采用与原生成路径一致的 execution decomposition；不能为了采用 endpoint 而忽略。
当 NInfer 自己生成的 accepted output 形成可由历史 renderer 精确重建的边界时，所有权链固定为：

```text
Frontend detects the model-output reconstruction boundary
  -> Engine transports relative accepted-prefix metadata
  -> Program atomically commits resident identity and shortlist digest
```

Frontend 只在 canonical boundary 恰好结束于 token frontier 时发布 metadata，并让 preview rollback、stop
truncation 和 speculative partial accept 服从同一个 accepted-prefix transaction。Program 把相对位置转换为
resident absolute frontier，并与 token ledger、State/KV 和 backend state 一起提交；它不扫描输出文本。
ResourceManager 不推断或放宽这个事实。这样未经客户端改写的 NInfer 输出可以继续采用 exact endpoint；客户端
重排 tool JSON、补默认值或改写历史仍是不同的 rendered identity，只能匹配更早的合法 checkpoint。

### 4.6 Request cache participation

每个请求在进入 Engine 前固定一种语义：

- `ReadWrite`：允许读取 exact candidates、capture checkpoints，并在 finish 时发布 continuation；
- `Disabled`：只允许 root materialization，结束时释放全部 active context resources。

`Disabled` 请求仍获得正常 active completion guarantee，但不会留下 private/shared catalog entry、
SessionIndex binding 或 Device/Host checkpoint replica。Serve warmup 使用这一语义。

---

## 5. State 与 KV 的 placement 语义

### 5.1 StateImage

StateImage 是完整迁移单位。设最大 active concurrency 为 \(C\)，额外 Device checkpoint slots 为 \(H\)，
则：

\[
Capacity_{DeviceState}=C+H
\]

前 \(C\) 份容量形成 active guarantee，全部 slots 仍由同一个 Program pool 管理，不与 lane 固定配对。
Host State 以完整 image 和独立 slot 容量计费。

State transition 的语义只有：

| 操作 | 条件与结果 |
|---|---|
| InPlace | 唯一 active writer 在自己的 destination 上推进 |
| Move | private source 被 consume，原 StateImage 成为 active destination |
| Fork | immutable 或仍需保留的 source 写入新的 private destination |
| Freeze | committed active StateImage 转为 immutable checkpoint |
| Snapshot/Restore | 完整 StateImage 在 Device 与 Host 之间复制 |

Program 根据完整 post-state 选择 Move 或 Fork。Fork source 在 destination 首次 state-writing commit 完成前
保持 pin；该结算属于 active model transaction，不能与新的 global resource transition 重叠。

Active State binding 同时记录 read source 的生命周期归属。Root、Move、Restore 和 active-capture Fork
的初始 read 由当前 sequence 提供；从 retained source 或仍有 surviving checkpoint reference 的 consumed
source 建立 Fork 时，read 由 surviving owner/reference graph 保留，active primary binding 只拥有
write destination。这个 borrowed read 参与全局物理占用并由 source pin 保护，但不能通过 primary binding
再次计费，也不能仅凭 primary binding 在 abort/release 时释放；若 surviving checkpoint 本身是 active
lineage 独占的 optional checkpoint，它只通过该 checkpoint ownership 计入 entitlement，并在该 ownership
结算后决定 source 是否可释放。Fork settlement 后 read/write 都切换到 destination，生命周期重新归属
active sequence。

### 5.2 Typed KV

每个 continuation 对 Main KV 及 selected backend KV 分别持有 typed address space。各 pool 的 frontier、
capacity 和 placement 独立。

资源规划只使用以下语义：

- required logical pages 与 committed coverage；
- 每页可用的 Device/Host replica；
- immutable references 和唯一 writer；
- partial-tail COW requirement；
- restore、demote 或 release 对 capacity 与 machine work 的影响。

具体 page geometry、Host packed extent、COW 和 block-table publication 由
[Paged KV Context Store](paged-kv-cache.md)定义。

Prefix Fork 后的 active address 可以按以下顺序同时包含共享与私有页：

```text
immutable full pages | unique mutable full pages | optional unique mutable tail
```

Active capture 对不可变整页只增加引用，对唯一可写整页原地 Freeze；非对齐尾页保留给 checkpoint，并只
复制一页作为新的 active writer。对齐 frontier 不发生 KV copy，后续增长再物化新的 writer page。
Program 的 capture assessment、reservation 和 commit 使用同一个 snapshot-shape 分析，因此资源信用、
transfer cost 与实际可提交的页所有权条件一致。

State 与 KV 可以独立 placement，例如 State 在 Device、Main KV 部分在 Device、其余 Main KV 在 Host，
backend KV 采用另一种 coverage。Checkpoint 只有在全部 required components 至少保留一份有效 replica
时才继续存在。

---

### 5.3 Persistent disk tier

Device checkpoint 与 pinned Host State slot 之上还有第三层：**Disk record**。它是可选的（`--prompt-cache`），
默认关闭，用于跨进程重启和超出 Host slot 预算的工作集。前两层丢失一个 checkpoint 后只能重新 prefill；
Disk record 让同一 prefix identity 可以以 I/O 代价恢复。

三层的语义差别是明确的：

| 层 | 归属 | 生命周期 | 恢复代价 |
|---|---|---|---|
| Device checkpoint | Program StateImage/KV pool | 进程内，受 pressure planning 支配 | 无或 D2D |
| Host State slot | Program pinned host store | 进程内，受 host 容量支配 | H2D |
| Disk record | `ContextDiskStore` | 跨进程，受 size cap 与 LRU 支配 | 文件读 + pinned staging + H2D |

Disk tier 不是 placement 的一种：它不参与 `Used_r(S)` 的物理容量计算，也不出现在 `ResourcePlan` 里。
它只在两个逻辑边界上起作用，且两处都不改变 Program 的物理 authority。

**身份。** Disk record 的键就是 checkpoint 的 `PrefixShortlistKey`（两个滚动 digest、frontier、
identity_tag），与 §4.5 exact identity 使用同一份内容索引。store header 另外记录 artifact 与 KV layout
signature；signature 不同的 store 被整体丢弃而不是部分信任，因为匹配它的任何一条 record 都会恢复出语义
错误的状态。

**Hook (a) — spill。** 两个触发点。Victim owner 在 `apply_private_action` / `apply_shared_action` 被
evict 时提交一次；但 victim 被 apply 时其 pages 可能已经释放，所以真正承载的是 terminal publication 处的
write-behind：checkpoint 刚进入 catalog 就提交，等到该 host slot 之后被 evict 时磁盘副本已经存在。
Device-only checkpoint 从不提交，回读会与 decode round 争用同一条 transfer stream。
提交只是入队；写入在专用 I/O 线程上按 batch 进行，decode round 不被阻塞。

**Hook (b) — lookup。** 在 candidate 枚举之前执行，而不是之后。只有当 disk index 中某个 frontier 长于
任何 resident 候选时才探测；命中则 restore 并装入一个 **空闲** catalog slot，随后按普通 resident source
参与 §7 的 target 搜索与定价，只有未覆盖的后缀进入 prefill。不会为此驱逐 resident owner：那等于用未经
验证的 disk source 换掉已证明可用的内存 source。把恢复出的字节变成 continuation 是 Program 的职责，
ResourceManager 通过 Engine 提供的 adoption 回调完成这一步。

**Hook (c) — startup。** store 打开时重放 journal，重建 record 与 extent 索引。没有启动期扫描或压缩，
因此启动代价与 record 数量成正比而与 pool 体积无关。

**Hook (d) — durability。** 发布顺序是：extent 字节先 flush，然后 journal 追加带校验和的 publish entry
并 flush。record 的存在性由该 entry 定义，所以断电或崩溃产生的残尾永远不会发布一条 payload 缺失的
record；重放在第一个不完整或校验失败的帧处截断。相应地，cancellation 只在任务开始前检查：字节已经写下
之后再放弃不会省下任何 I/O，只会在 pool 里留下无人引用的 extent。

**存储格式。** 一个 cache 目录两个文件：`blobs.dat` 是 append-only、4 KiB 对齐、按内容寻址的 extent 流；
`index.log` 是 append-only journal（sector 大小的 header，随后是长度与校验和分帧的 entry）。按内容寻址
意味着后一轮与前一轮共享的 page 只写一次，所以每轮 spill 是小增量而不是整份重写。选择 segment + journal
而不是每 record 一个文件，是因为 77k token 的 record 约一千多个 KV page：每 page 一个文件输给一次顺序
extent 流，每 record 一个文件则丢掉跨轮共享。

**容量与回收。** `--prompt-cache-max-bytes` 是 live extent 字节上限。触顶时一次性按 LRU 回收到低水位
（默认 75%），而不是每次发布驱逐一条，避免逐轮抖动。被回收 record 的 extent 变成 dead bytes，由
mark-and-sweep 压缩回收；压缩同时受 dead extent 数、dead 字节和碎片率三个阈值约束，不会为少量死页重写
多 GB 的 pool。压缩在 I/O 线程上进行，复制阶段不持 index 锁，只在最后的文件交换时短暂持有，因此不阻塞
executor；复制阶段可以被新到达的请求取消，此时活动 store 完全未被触碰。

**批量。** 一条 record 的 page 数可达上千，整份 staging 会在 24 GB 级显卡上直接 OOM。spill 与 restore
都按 `ContextDiskTransferPort::batch_pages()` 分批，tier 同时持有的 page 不超过一批。

**Signature 组成。** store header 的 signature 由 Program 组成（`context_cache_signature.*`），因为
只有模型知道哪些事实会改变一页字节的含义。`Program::context_cache_signature(artifact_identity)` 取
Engine 在装载时已经算出的 `LoadSummary::prefill_signature` 作为 artifact 半边（它包含架构、config 解释
与完整的绑定权重清单及其存储格式，很长，所以只有它的 64-bit FNV-1a 摘要进入 signature），再加上：

| 组成项 | 来源 | 为什么必须包含 |
|---|---|---|
| `a<16 hex>` | `LoadSummary::prefill_signature` 摘要 | 换一个 artifact 或换一种权重格式后，同一段 prefix 的 KV 不再等价 |
| `kv<spec>` + `t<tag>` | `KvCacheSchedule` 的 spec 与 `identity_tag()` | 一种 KV 表示写下的页不能按另一种读；`identity_tag` 与 §4.5 内存内 prefix identity 用的是同一个值 |
| `sb`/`ph` | `SpeculativeBackend`、`ProposalHead` | backend 拥有第二个 KV pool，DFlash 还占 State image 的一部分 |
| `c`/`k` | `max_context`、解析后的 KV capacity | 决定 page 数与 entitlement |
| `mp`/`mn`/`mb`、`bp`/`bn`/`bb` | 两个 pool 的 `KVPageGeometry::page_tokens`、plane 数与 `plan_host_kv_page_layout().page_stride` | record 的一页就是一个 page group 的 `page_stride` 字节；per-layer schedule 让每层 plane 字节数不同，`page_stride` 是唯一覆盖全部层的数字 |
| `s` | `StateImageDevicePool::host_layout().image_bytes` | State image 的字节布局 |
| `v` | vision 是否启用 | 影响 position 与 prefix identity |

首段 `q35-ctx1` 是格式版本：改变任何一项的拼写方式都必须提升它，否则两种不同的组成可能拼出同一个
字符串并互相匹配对方的 record。整串只使用目录名可以原样保留的字符，所以 store 目录名不会把两个不同
的 signature 折叠成同一个目录。

**当前状态。** store、journal、LRU、压缩、取消、options 解析、signature 组成与 stats 已经就位；
capture 一侧的 port 按上面的快照模型实现。把一条恢复出来的 record 变成 catalog 里的 continuation
仍然缺一个 Program 事务：`KVAddressSpaceStore` 只能在 active address space 上 `ensure_mapped_to_tokens`
并 `commit_frontier`，没有从既有页直接装配一个 inactive address space 的入口，而 continuation 还需要
ledger、`ResidentPrefixIdentity`、rope delta 与 execution frontier 才能通过 exact verification。因此
record 的 State image 除了 pinned StateImage 负载之外还要带一个 Program 自己定义的头部（ledger、
prefix identity、frontier），`ContextDiskStore` 对此保持 payload-agnostic。在该事务落地之前，
`ResourceManager::attach_disk_tier` 的 adoption 回调返回 disengaged，命中的 record 被释放、请求回退到
prefill。

**Transfer port 的线程契约。** `ContextDiskTier` 自带 I/O 线程，`begin_capture` / `capture_pages` /
`capture_state_image` / `end_capture` 都在该线程上运行，与 Engine worker 的 decode round 并发；而
`lookup` / `restore` 运行在调用线程，也就是 Engine worker 自己。ProgramImpl 是单 mutation owner
（见 [Engine 架构](engine-architecture.md#51-单一-mutation-owner)），不是线程安全的，所以 port 的
capture 一侧**不能**在 I/O 线程上触碰 ProgramImpl：

- Engine worker 在 offer 时就把 checkpoint 的 host replica（pinned Host State slot 与 Host KV
  extent）钉住并登记成一份快照，然后才入队 spill；
- I/O 线程上的 `begin_capture` 只是在 port 自己的互斥量下查这份快照，`capture_pages` 只是从已经
  pinned 的 host 内存 memcpy 进 staging，不发起任何 CUDA 调用，也不修改 Program 状态；
- `end_capture` 只标记完成，实际解钉在 Engine worker 的下一次维护步里做。

由此得到一条策略：只有 **State 与 KV 都已经 host-resident** 的 checkpoint 才会被 offer。Device-only
的副本要回读就得占用 decode round 正在用的 transfer stream，这正是 §5.3 开头拒绝它的理由；把同样的
规则扩展到 KV 页，capture 一侧就完全不需要 transfer stream 与 event 同步。restore 一侧运行在 Engine
worker 上，因此可以直接使用 Program 的 `cudaMemcpyAsync` + `CudaCompletionEvent` 路径把字节放进
Host State slot 或 device 页。

**批量与 per-layer 几何。** record 的一页是一个 page group，字节数取 `plan_host_kv_page_layout` 的
`page_stride`；per-layer KV schedule 让每层 plane 的字节数不同，但一个 page group 覆盖所有层，所以
`page_stride` 是唯一正确且统一的 page 字节数。绝不要用固定 page 大小推算偏移。

**DirectStorage。** Windows 上可以用 kernel-bypass DMA 跳过 pinned staging 往返。seam 声明在
`ContextDiskTransferPort` 中并由 `_WIN32 && NINFER_DIRECTSTORAGE` 限定，本仓库不实现它（需要 Windows SDK、
D3D12 设备和真实 GPU）。实现时必须在释放任何 staging 资源之前于 CPU 线程等待共享 D3D12 fence，否则
DirectStorage 队列仍有请求在飞时释放 COM 资源会在 NVIDIA D3D12 驱动内崩溃。

---

## 6. Active completion guarantee

### 6.1 Admission reservation

一个 candidate 发布为 Active 前，Program 为它保障：

```text
required Device checkpoint prefix
+ remaining prompt growth
+ maximum effective output growth
+ target-defined provisional/speculative growth
+ State and partial-tail COW destinations
+ selected backend requirements
```

这些资源绑定在返回的 `SequenceHandle` 中。物化进度只在 allocation 与 reserved-but-unmapped 之间转换，
不会把容量交给另一个 request。

Active truncate 或 speculative rollback 可以解除 mappings，但对应容量仍属于该 active reservation。
只有 terminal release 或明确缩减 active entitlement 的资源 transition 才能把容量归还全局。

Active entitlement 只包含 active owner 的 destination、私有增长 reservation 和已归属于该 lineage 的
exclusive optional resources。Fork 期间借用的 immutable StateImage/KV source 不能通过 primary binding
再次计费：它若由其他 surviving owner 保留则只存在于全局 physical occupancy，若由 active lineage 的
optional checkpoint 独占则只通过该 checkpoint 计一次。否则同一 allocation 会被重复收费，并把合法的
Fork 错判为超出 active guarantee。

### 6.2 Terminal 与 capture

TerminalPending request 继续持有 `SequenceHandle` 和完整 reservation，直到 Program 完成：

- **Finish**：发布一个完整 immutable checkpoint；或
- **Discard**：释放整个 active continuation。

若 Finish 无法形成合法保留终态，必须采用 Discard。Optional capture 可以通过同一 pressure graph 改变
inactive owners，但只有严格优于 private-only baseline 的完整终态才会提交；否则跳过 shared publication，
不阻塞 active request，也不改变同一 frontier 原本可执行的 private capture。

### 6.3 Persistent backfill proof

Scheduler 是否允许 backfill 由 [Engine 架构](engine-architecture.md#52-admission-顺序)决定。Program
提供的物理证明使用同一 feasibility oracle：

```text
hypothetical state =
    current stable state
    - protected donor active reservations after their terminal release
    + borrower full active reservation
    + head root materialization
    with all unprotected inactive cache releasable
```

只有该完整状态可行，borrower 才具有 persistent-safe backfill 资格。证明与 `resource_revision` 绑定，
不使用 borrower 或 donor 的预计完成时间。

---

## 7. Materialization planning problem

### 7.1 Candidate

ResourceManager 为已选 request 建立有界 candidate 集合：

- root；
- matching private endpoint；
- typed rewrite checkpoint；
- retained long anchor；
- shared stable prefix。

相同 capability 经不同 index 命中时先去重。Program 对每个 shortlist entry 执行 exact verification，
并产生一个 `AdmissionCandidate`。Candidate 固定 source、destination、prompt 和 request work，但不提前
决定 Move/Fork、victims 或最终 placements。

Root 始终存在。它不复用 continuation state，代表从 prompt 开始正常 prefill。

### 7.2 Shared publication candidate

Shared prefix 的三个状态不能混为一谈：

- **boundary** 是协议或 Engine 指出的逻辑内容边界；
- **candidate** 是 Frontend 已解析到 exact token frontier 的可选写入机会；
- **owner** 是 Program 已发布、具有完整 State/KV identity 与 placement 的 immutable checkpoint。

外部协议或 C++ `PromptInput` 每个请求最多提交四个显式 markers。Frontend 还可以生成最多三个 Engine
candidates：全部 tools 之后、连续 leading System/Developer 之后，以及 full prompt。相同 frontier 合并，
因此每个请求最多七个 prepared candidates。这个固定上限不是启动配置。

Shared catalog 是 Engine-wide 公共容量，不是每条 lineage 的配额。启用 context cache 时，默认 logical
capacity 同时覆盖 active concurrency 下限和单请求最多四个显式 markers，即
`max(max_concurrency, kMaximumExplicitPromptCacheMarkers)`；显式配置仍完整覆盖默认值。这个下限允许较早的
稳定层与较晚的滚动 marker 同时成为 owner，但是否 capture、保留或替换仍只由通用 portfolio/pressure
planning 决定，不提供 Claude、compact 或 token-position 特例。

Candidate 保存 evidence flags。策略含义为：

| Evidence | 首次出现时的准入能力 |
|---|---|
| `ExplicitBoundary` | 可以参与 pressure，但仍须有严格正净收益 |
| `RequestedAutomatic` | 可以参与 pressure，但仍须有严格正净收益 |
| `DefaultAutomatic` | 只能使用不降低现有 owner 的空余终态 |
| `EngineStructural` | 只能使用不降低现有 owner 的空余终态 |
| `EngineObserved` | 至少两个独立 reuse domains 观测到相同 key 后才可创建 |

Marker、evidence 和 shortlist key 都不证明命中；Program 对 read、dedup 和 publication 重新验证完整
identity。候选创建 source 的顺序为：exact shared owner 直接 dedup；selected exact private reuse base 在
第一次模型 mutation 前做零-prefill promotion；更晚 frontier 由 active prefill capture。早于 selected
reuse base 且没有现成 exact source 的候选只保留为需求观察，不回滚状态或重新 prefill。

Shared candidate 必须在 prefill 前决定是否增加 split，而物理状态只能在 frontier 到达时确定。因此实现
分为两步：

1. admission 已选定 materialization source 后，ResourceManager 用完整逻辑 portfolio 的乐观收益上界和
   Program 返回的精确 split cost 选择 candidate subset；这一步不预留资源，也不承诺 publication；
2. capture offer 到达后，`SharedCapturePlanner` 从当前 `resource_revision` 搜索完整物理 targets，可以联合
   demote/evict 多个合法 inactive private/shared owners。shared catalog 已满时，logical replacement 与其他
   物理 victims 属于同一个 target 和同一个 transaction。

实际 planner 的 baseline 是同一 frontier 的 private-only policy。Shared publication 是可选投资：没有可行
且严格正收益的 target 时执行 private baseline 或 Skip。

### 7.3 Target

Planner 的搜索节点是完整 target，而不是一条孤立 eviction action。一个 target 同时确定：

```text
selected candidate
selected source retained or consumed-to-active
every eligible owner retained or evicted
every surviving checkpoint and required coverage
every surviving State/KV Device/Host placement
logical publication destination
```

Eligible victims 只包括未被 active reference、selected-source protection 或 open transaction pin 保护的
inactive owners。Target 应用后产生的零引用 objects 自动释放。

Program 从该完整终态重新推导：

- Move 或 Fork；
- State/KV replacement 与 partial-tail COW；
- last-reference releases；
- active completion reservation；
- Host allocator geometry；
- ordered stage peaks；
- immediate work 与未来 recovery work。

一个 candidate 的 `identity target` 只应用 incoming activation 与必需的 source disposition，保持其他
eligible inactive owners 的 checkpoint contents 和 placements 不变。它是该 candidate 压力降级图的根。

### 7.4 单调降级图

Pressure target 从 candidate 的 identity target 沿单调降级方向扩展：

- 移除冗余 Device 或 Host replica；
- 将 Device State/KV 降为 Host-only；
- 缩短不再保留的 checkpoint coverage；
- 删除 checkpoint；
- evict 完整 inactive owner。

Program 根据当前 residual deficit、checkpoint frontiers、page/extent boundaries 和共享引用生成语义
successors。搜索不枚举物理操作的排列；每个 target 由 Program 映射到唯一规范 stage order。

单个 owner action 暂时不产生 reclaim，也不能在联合 post-state 之前被删除，因为它可能与另一 action
共同形成 last-reference release。

---

## 8. 规划目标与算法

本节中的 `incumbent` 指当前已经找到的最佳可行完整 target。

### 8.1 可行性与价值分离

Program 先判断 target 是否结构合法、物理可行以及是否仍可继续降级。ResourceManager 只在物理可行且
logical publication 可采用的 targets 之间比较价值。

成本模型不参与容量判定。任何预测误差都只能改变“选择哪个可行方案”，不能让不可行方案通过。

### 8.2 机器成本

Program 将物理工作折叠为：

```text
Immediate =
    ordered transfer phases
  + remaining Text/Vision prefill
  + required State/KV copy work
```

一次同 phase、同 direction 的 transfer batch 使用：

\[
Transfer =
\max(batch\_ns+operations\cdot operation\_ns,\ bytes\cdot ns\_per\_byte)
\]

Text suffix 的 prefill work 使用：

\[
AttentionPairs = B\,S+\frac{S(S+1)}{2}
\]

其中 \(B\) 是已复用 prefix tokens，\(S\) 是剩余 suffix tokens。Vision item/patch work 使用同一
startup-resolved machine model 的独立分量。

Transfer 系数按硬件选择，prefill 系数按硬件与实际 Text/Vision config、绑定结构和 Use 生成的
`prefill_signature` 选择。签名不使用 checkpoint/release 名称，也不包含权重数值。两组系数独立
解析；没有匹配的标定时使用对应的通用系数。成本只影响可行方案之间的排序，不是加载或执行门槛。

### 8.3 Portfolio value 与 future loss

ResourceManager 保存最近 32 个成功 materialize 为 Active 的请求。每条 demand record 保存 reuse domain
以及该请求 exact-matched、selected 或可创建的 prefix keys；不保存 prompt、媒体内容、session 原文或 API
cache key。有 Engine session key 的请求使用其稳定 digest 作为 domain；无 session key 的每个请求形成独立
domain。

对 checkpoint \(p\) 和完整 portfolio state \(S\)：

\[
Saving(p,S)=\max(0,\ Rebuild(p)-Recovery(p,S))
\]

`Rebuild` 是从 root 到该 frontier 的 canonical prefill cost。Program 根据实际 State/KV placement 枚举所有
受支持的 restore/copy/interval-prefill recovery recipes；common runner 用本次 planning problem 的同一个
immutable machine cost model 定价并取最小值作为 `Recovery`。Program 不接收 cost model，也不替策略层选择
recipe。每条 demand 只取所有 matching checkpoints 中的最大 saving：

\[
EmpiricalValue(S)=\sum_q\max_{p\ matches\ q}Saving(p,S)
\]

因此 tools、instructions 和 full-prompt 等嵌套 prefixes 不会对同一个未来请求重复计价。再加上尚未到期
的显式 shared credit，得到公共价值：

\[
PublicValue(S)=EmpiricalValue(S)+SharedCredit(S)
\]

Private retention 表示每个 owner 还有一次未观测的后续复用。比较 baseline \(S_b\) 与 target \(S_t\) 时，
保持 checkpoint identity 不变，先计算每个 checkpoint 的 saving 损失，再在 owner 内取最大值：

\[
PrivateLoss_o(S_b,S_t)=w_o\max_{p\in o}
\max(0,Saving(p,S_b)-Saving(p,S_t))
\]

不同 owners 的 `PrivateLoss` 相加。这样不会把同一 continuation 的嵌套 checkpoints 全部当作未来请求，
也不会让一个仍然可用的末端 checkpoint 掩盖较早 TurnClosure 或 long anchor 的损失。权重为：

| Private retention | prior weight |
|---|---:|
| Disposable | 1 |
| RecentPrivate | 4 |
| LiveSession | 16 |

Shared owner 没有固定 retention multiplier。`ExplicitBoundary` 或 `RequestedAutomatic` 在 publication 时带来
一个 owner-scoped credit；它在第一次后续 exact match 时消费，或在 32 次成功 materialization 后到期。
同一 frontier 的多个 evidence 不叠加 credit。

对 mandatory materialization 的 candidate identity state \(I_c\) 和 pressure target \(T\)：

\[
FutureLoss_c(T)=\max(0,PublicValue(I_c)-PublicValue(T))
+\sum_o PrivateLoss_o(I_c,T),\qquad
J(c,T)=Now(c,T)+FutureLoss_c(T)
\]

对 optional shared capture，比较 private-only state \(S_b\) 与 complete shared target \(S_t\)：

\[
NetGain=PublicValue(S_t)-PublicValue(S_b)
-\sum_o PrivateLoss_o(S_b,S_t)-ImmediateDelta
\]

执行前 candidate subset 另减精确 `SplitCost`；capture offer 到达后 split 已经发生，不再重复计费。
`ImmediateDelta` 包括 shared target 相对 private baseline 的 transfer/copy cost，并乘以当前 capture owner
及被该 global transition 阻塞的其他请求数。只有 `NetGain>0` 才发布；相等、溢出或饱和无法证明严格
正收益时保持 baseline。

Selected source 的合法 `ConsumedToActive` 是 ownership transfer，不计作 eviction。其他随 pressure
删除或降级的 checkpoints 通过 public value 与 private transition loss 进入 future loss。

### 8.4 排序提示不是证明

Program 为未 assessment 的 target 返回 physical residual、预计剩余步数、transfer/recovery work 和
稳定 ordinal。ResourceManager 将这些事实与 portfolio policy、logical publication 条件合并成 priority。
构造目标同时包含物理容量和逻辑可采用性；物理缺口为零不能代替 publication slot 检查。

这些值只决定探索顺序及 optional 额度边界是否值得继续搜索：不能标记 target feasible，不能证明
incumbent 最优，不能参与 physical readiness。组合动作可能取消拷贝，预测成本不是可用于剪枝的
数学下界。只有完整 exact assessment 与 logical-adoption check 可以产生可采用方案。

### 8.5 无压力路径

Planner 必须先读取全部 exact candidate identities。只有 Program 明确表示所有 candidate 均不需要也不能
通过 pressure 改善当前 materialization work 时，才直接 seal 最佳可采用 identity。只要任一 candidate
仍 expandable，就建立同一 bounded pressure domain；不能因为估计成本高于 incumbent 而跳过它。

最终 shared-capture split 由 ResourceManager 在 selected candidate 已确定后选择。Program 在 seal 内验证并
消费 `FinalScheduleIntent`；返回的 `ResourcePlan` 不再具有 post-seal mutation API，Scheduler 看到的
summary 已是最终值。

### 8.6 Correctness incumbent

若没有 identity target 同时满足物理与 logical publication 条件，Planner 在 optional budget 之外 assessment：

```text
root candidate
+ release all unprotected inactive cache
```

root maximal target 由 candidate-specific eligible domain 构造；selected source 没有 victim cell。它不保留
可牺牲 cache value，但保证 heuristic、target budget 或 wall budget 失败不会把本可运行请求误判为 blocked。
active requirements、source required coverage、writer 和 transaction pins 始终留在 Program 的 requirement
union 中。

若 isolated root 超过静态合同，结果为 permanently infeasible；若 isolated root 可行但 active guarantee、
lane 或 open transaction 阻塞，结果为 temporarily blocked。

### 8.7 有界 heuristic search

Materialization 与 shared capture 使用两个 typed entrypoint。Materialization 的 incumbent 是已验证
identity 或 root maximal；shared capture 的 incumbent 是 Skip，只有 exact `NetGain>0` 才替换。

一次 planning problem 中：

1. ResourceManager mint candidate/owner IDs，保留到 catalog capability 的唯一映射；Program 为每个
   candidate 建立 selected source 结构性缺席的 eligible victim domain。
2. Program 持有 opaque targets 与分片构造状态；Runtime 按预计完整 J 和缺口改善排列合法选项，
   兼顾价值与尽快形成可行方案，不构造 State/KV action。
3. 优先评估完整候选，失败后根据精确物理与逻辑反馈继续修复。Host geometry 等不能由总字节量
   证明的条件需要更早的 exact feedback。局部构造以外保留普通 alternatives。
4. 完整联合 post-state 结算 unique objects、alias、Move/Fork、COW、Host geometry 与 stage peak。
   只有物理可行且 logical adoption 成功的结果参与最终 J 与稳定 tie-break 比较。
5. 任意 optional 时间、工作或容量预算耗尽，返回最佳已验证 incumbent。

构造、修复、普通 expansion、maximal target 与 seal 共用 candidate-specific domain 和 exact evaluator。
游标、target arena 与借用摘要只在同一 planning session/revision 内有效；批次停止或丢弃不能产生
真实资源 mutation。结束搜索后不跨 decode/prefill 恢复旧状态。

Engine 提供 admission boundary 共享的 optional 时间额度，head/backfill inspections 不能各自重置。
预算考虑其他 runnable 请求及本 boundary 已花费的时间。Planner 先用短窗口搜索，再根据尚可获得
的预计改善和完成成本决定是否续额；不完整预测只有受限探索机会，已实现收益不能反复用于续额。
Target、工作量和 session storage 仍独立有界。

Identity 与 root maximal correctness assessment 不因 optional 额度耗尽而取消。时间在 Program
operation 之间检查，估计误差可造成单步越界；最终 seal 仍必须完成。因此预算只影响 cache quality，
不改变 mandatory readiness，也不承诺整个 planning 调用可在任意时刻中断。
Shared capture 维持独立的固定 target budget 与 Skip incumbent。

### 8.8 目标函数与确定性

对 exact feasible assessment：

```text
J = price(Program machine work)
  + ResourceManager portfolio loss
```

Program 只形成 canonical transfer/prefill work，并为 checkpoint recovery 枚举受支持的物理 recipes；
ResourceManager 提供 demand、retention 和 shared credit；runner 使用同一次 planning problem 的 immutable
machine cost model 统一定价并比较。成本模型不会进入 Program API，只能改变已验证 targets 的探索和选择
顺序，不能改变 physical status、readiness、source protection 或 target generation。

相同 `J` 下依次偏好：

1. 少影响已观测命中的 checkpoints，保留最近命中；
2. 少 owner eviction、checkpoint deletion、copy operation 和 transferred bytes；
3. 少剩余 Text/Vision prefill；
4. 多复用 prompt tokens；
5. 当前 session binding；
6. candidate 与 target 的稳定 ordinal。

停止原因只描述实际边界：`no_pressure`、`queue_exhausted`、`target_budget`、
`expansion_capacity`、`time_budget`、`work_budget` 或 `insufficient_expected_gain`。
`time_budget` 也包括预计下一不可分操作无法放入余量的情况。它们不声明当前 target graph 或真实
TTFT 的全局最优性。

### 8.9 Readiness

最终结果只有四类：

| 结果 | 含义 |
|---|---|
| `PermanentlyInfeasible` | isolated root completion requirement 超过当前 Engine 合同 |
| `TemporarilyBlocked` | 单请求可行，但当前 active/lane/transaction 状态阻塞 |
| `NeedsTransfer` | target 可行，需要异步 transfer 或 pressure stages |
| `Ready` | target 已 Device-ready，可在当前 boundary 同步发布 |

`TemporarilyBlocked` 只在 lane、FIFO protection、open transaction 或 resource revision 变化后重新检查。
普通 decode 不重复运行 planner。

---

## 9. Resource transition

### 9.1 三个对象

需要 allocation、pressure 或 transfer 的终态通过：

```text
ResourcePlan
  -> RunningTransaction
  -> ResourceResult
```

- `ResourcePlan`：Program seal 的 immutable、move-only、revision-bound transition；
- `RunningTransaction`：Program 当前执行中的唯一 global resource transition；
- `ResourceResult`：Committed 或 Aborted 的完整稳定结果。

Active reservation 已经覆盖且不需要新 allocation/transfer 的 finish、discard 或 inactive release 可以作为
零阶段原子 transition 完成，但仍遵守同一串行化、revision 和完整结果规则。

### 9.2 Start

Program 在第一次 mutation 前重新验证：

1. plan 的 `resource_revision`；
2. source、victim 和 destination capability generation；
3. ResourceManager 已保留的 logical publication capacity；
4. 当前 allocator 对每个 ordered stage 的容量和几何；
5. source/victim dependencies 与 active protections。

Stale 或不再可行的 plan 在 start 前无物理副作用。Start 成功后，source、victims 和 destinations 被
claim/pin，plan 被线性消费，不再更换 candidate 或 target。

ResourceManager 在调用 Program start 前先建立唯一 logical transaction record，并把所有 source/victim 与
publication destinations 置为 claimed/reserved。Program 在第一次物理 mutation 前完成自己的 preflight。
同步 start 返回 Aborted 时，ResourceManager 回滚完整 logical claim；start 成功后 logical record 与 Program
transaction 一直同时存在，直到 stable result adoption 或 Engine-wide cleanup。

### 9.3 Progress

Transaction 可以跨同步 copy 或 CUDA event 推进。每个 replacement 遵循：

```text
destination reserved
  -> copy complete
  -> epoch/coverage verified
  -> replacement published
  -> old replica released when unreferenced
```

Materialization 与 active capture 共用一条单调 pressure phase：

```text
HostReleases -> CopyPreparation -> CopiesInFlight? -> CopyPublication -> Committed
```

phase 是全局 pressure 顺序的唯一 authority。每个 owner work 可以保存 destination/copy 是否已实际提交以及
mutation 是否已发布的执行事实，用于形成 absolute result；这些事实不能改变或绕过全局 phase。无 copy 时从
`CopyPreparation` 直接进入 `CopyPublication`。Cancellation 只在稳定 phase boundary 采用，已发布的 Host
release 或 victim eviction 通过绝对 `ResourceResult` 报告。

同一时间至多一个 global resource transition。既有 active execution 只有在使用自身既有 reservation、
且不触碰 transaction mappings 时才能与 transfer 交错。

### 9.4 Commit

Commit 发布：

- new active sequence 或 checkpoint；
- selected source disposition；
- 已提交的 owner/checkpoint degradation；
- stable State/KV placements；
- transaction pin/reservation cleanup；
- 必要的 `resource_revision` 变化。

`ResourceResult` 包含 ResourceManager 采用逻辑终态所需的绝对结果，不包含供上层重建物理 occupancy 的
delta。每个 owner claim 绑定确切的 `CheckpointRef` 删除集合，而不是只绑定删除数量；ResourceManager
逐项核对最终 summary 恰好等于原 checkpoint 集合减去该集合。因此同数量但不同 checkpoint 的替换会在任何
catalog mutation 前失败。ResourceManager 在 start 前已预留 adoption storage，因此 commit 后的 adoption
不能进行新的容量判断。

### 9.5 Abort

Publication 前的 cancellation 或执行失败可以得到 request-local abort，只要 Program 仍能形成稳定物理状态。
Abort 保证：

- selected source 仍有效；
- target 没有半 Active 或半 published checkpoint；
- unused destinations、reservations 和 pins 已释放；
- 已安全完成的 victim demotion/deletion 被明确记录在 `ResourceResult` 中。

已经发布 replacement 后完成的合法 pressure degradation 不需要逆向复制回原 placement。ResourceManager
采用 abort result 中的实际终态，而不是回放中间 receipts。

若 Program 已无法形成可解释的稳定状态，则进入 Engine-wide failure。

---

## 10. Retention、session 与 publication

### 10.1 Retention policy

Private retention class 通过 owner prior 影响 portfolio value，不提供固定 eviction 顺序。Shared owner 的价值
来自真实 demand 与尚未到期的 explicit credit。Planner 可以组合：

- 删除冗余 replica；
- 将 Device replica 降为 Host-only；
- 删除不再保留的 checkpoint 或 suffix coverage；
- evict 完整 inactive owner。

如果一个 placement change 会删除 checkpoint 的最后一份完整 State 或 required KV coverage，同一个
target 必须先建立 replacement，或同时删除该 checkpoint。

Placement 只在 admission、capture、finish 或显式 inactive release 的 resource boundary 改变。普通 decode
不运行周期性 promotion/demotion。

### 10.2 Session ordering

SessionKey 属于 ResourceManager，只提供 candidate lookup 与 binding。Program 不读取 SessionKey。

每个可更新 SessionIndex 的请求取得单调 `publication_order`：

- 新结果只有 order 更大时才能替换当前 binding；
- 较旧请求晚完成时成为 anonymous cache 或按 policy 释放；
- matching private source 可以在合法时 consume-to-active；
- shared source 始终保持 immutable；
- abort 只能恢复自己仍持有 exact claim 的 former binding。

Capability generation 判断 handle 是否仍有效，publication order 判断哪个完成结果更新 session；两者职责不同。

### 10.3 Logical publication capacity

ResourceManager 必须在 Program mutation 前预留 active/private/shared destination slot。Program commit 后的
logical adoption 不再扩容或搜索 slot。

Private Move 可以复用 source descriptor；root 或 Fork 需要 destination。Optional capture 没有 slot 时跳过；
terminal Finish 没有合法 publication capacity 时采用 Discard。

---

## 11. 配置语义与有界性

`ContextCacheOptions` 在 Engine 构造时解析为固定容量：

| 配置轴 | 架构含义 |
|---|---|
| `device_state_slots` | active guarantee 之外可保留的 Device checkpoint StateImages |
| `host_state_slots` | 完整 Host StateImages |
| `host_kv_capacity_bytes` | typed packed Host KV arena 总容量 |
| `max_private_continuations` | private owner/catalog 容量 |
| `max_shared_prefixes` | shared immutable owner/catalog 容量 |
| `max_long_anchors_per_continuation` | 每条 private history 的 retained long checkpoints 上限 |

`PromptCacheOptions` 独立解析第三层（§5.3）：

| 配置轴 | 架构含义 |
|---|---|
| `enabled` | 是否启用 Disk tier；关闭时前两层语义完全不变 |
| `directory` | store 目录；空值取 `<artifact dir>/.ninfer-cache/<config signature>` |
| `max_bytes` | live extent 字节上限，超出按 LRU 回收到低水位 |

所有 stores、catalogs、Program unique-object scratch 和 reusable planner frontiers 都按解析后的上限建立。
每个 pressure planning session 的 target arena、hash 和 assessment storage 在 session 开始时按固定上限取得
容量；target expansion 不得突破该容量。Logical claim manifests 与 adoption 所需容量必须在 Program mutation
前取得。每请求 external markers 固定最多四个、Frontend candidates 固定最多三个、demand window 固定为
32；这些是产品语义与算法有界性，不是可调容量轴。
配置必须满足：

- private continuation capacity 至少覆盖全部 active requests；
- Device State 总容量为 `max_concurrency + device_state_slots`；
- Host State 与 Host KV 独立计费；
- logical counts、address-space counts 和 storage sizes 可表示；
- Program 可以兑现最小 active capacity 与 selected backend requirements。

Context cache disabled 时采用 root-only 语义：不读取或发布 inactive continuation，Device/Host checkpoint
容量为零，但 active execution 所需的 State/KV reservation 保持不变。

公开默认值和启动命令由 `EngineOptions`、[CLI](../cli.md)与 [Serving](../serving.md)维护，不在本架构
文档中复制。

---

## 12. 核心不变量

0. Disk record 只在其 publish entry 完整落盘后存在；它不参与物理容量计算，也不能替代任一物理 store 的
   residency 查询。
1. Scheduler 先选择 request；资源层不能以 cache value 改变请求顺序。
2. ResourceManager 是 logical cache policy 的唯一 authority，Program 是 physical state 的唯一 authority。
3. Physical occupancy 按 unique allocations 与 concrete reservations 计数，不按 logical references 重复计数。
4. Reclaim、Move/Fork 和 COW 必须从完整联合 post-state 推导。
5. 每个有序 transition stage 都必须满足容量与 allocator geometry。
6. Prefix hit 必须具有 exact identity、完整 StateImage 和全部 required typed KV coverage。
7. Published checkpoint immutable；每条 active continuation 至多一个 mutable writer。
8. Active completion reservation 在 terminal resource result 采用前不可被借用或回收。
9. 同一时刻至多一个 global resource transition，且 sealed plan 与 `resource_revision` 绑定。
10. Start 前 stale rejection 无物理副作用；start 后不能更换 candidate 或 target。
11. Program commit/abort 都返回完整稳定终态；ResourceManager adoption 已预分配且不能失败。
12. Root 加释放全部 unprotected inactive cache 是 bounded search 之外的 correctness fallback。
13. Planner 成本模型只排序可行 targets，不参与物理正确性。
14. Session binding 只接受更大的 publication order。
15. Retention 或 capture 失败必须退化为 skip/release，使 active lane 有有限终态。
16. Shared publication 必须严格优于 private-only baseline；单个 demand 对嵌套 prefixes 只能贡献一次最大
    saving。
17. Candidate selection 不是资源预留；实际 capture target 必须在 frontier 到达后按当前 revision 重新
    证明完整物理终态。

---

## 13. 实现位置

| 职责 | 主要位置 |
|---|---|
| logical catalog、claims、session policy | `src/runtime/engine/context_cache/resource_manager.h` |
| bounded target ledger 与 common search primitives | `src/runtime/engine/context_cache/resource_search.h` |
| cross-candidate bounded planner | `src/runtime/engine/context_cache/materialization_planner.h` |
| typed shared-capture bounded entrypoint | `src/runtime/engine/context_cache/shared_capture_planner.h` |
| shared/private portfolio value | `src/runtime/engine/context_cache/context_portfolio_value.h` |
| machine cost model | `src/runtime/engine/context_cache/context_cost.cpp` |
| prefill measurement signature | `src/models/qwen3_5/measurement.cpp` |
| common resource summaries | `src/runtime/contract/resources.h` |
| unique-object projection contract | `src/models/qwen3_5/program/planning/resource_projection.h` |
| target domain、projection 与 transaction | `src/models/qwen3_5/program/planning/`、`src/models/qwen3_5/program/transactions/` |
| State stores | `src/models/qwen3_5/program/storage/state_store.h` |
| logical KV/address spaces | `src/models/qwen3_5/program/storage/kv_store.h` |
| Host KV extents | `src/models/qwen3_5/program/storage/host_kv_store.h` |
| persistent tier policy、spill/restore 状态机 | `src/runtime/engine/context_cache/context_disk_tier.h` |
| on-disk 格式、journal、LRU 与 mark-and-sweep | `src/runtime/engine/context_cache/context_disk_store.h` |
| positional/O_DIRECT 文件原语 | `src/core/positional_file.h` |
| public capacity options | `include/ninfer/types.h` |

路径用于定位当前实现，不改变本文定义的所有权边界。
