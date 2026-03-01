# 第 6 章：喂数这件事（Parquet→Token→(B,T)、DDP 分片、BOS 对齐 best-fit、断点续训）

预训练的 GPU 集群像一座座高速运转的“算力熔炉”，而 DataLoader 则是把燃料按节拍送进去的传送带。传送带慢一点，表现就是 MFU 断崖式下跌；传送带“喂错了”，表现就是 loss 不收敛、评估指标异常、或者模型生成出现诡异的跨文档串味。更糟的是：DataLoader 的错误往往是静默的——你不会立刻报错，但你会在几小时后发现曲线已经偏离基线。

本章把 nanochat 的“喂数链路”讲透：从 `NANOCHAT_BASE_DIR/base_data/` 里的 Parquet 文本，到 GPU 上的 `(B, T)` 整数张量；从 DDP 下的 row group 分片，到 `state_dict` 驱动的断点恢复；以及最关键的一点——nanochat 的 packing 不是传统的“块对角注意力掩码”，而是用 **BOS 对齐 + best-fit** 让链路极简且能跑在原生注意力内核上。理解这些细节，你才能在遇到“慢/抖/不对齐”时快速定位根因。

---

## 6.1 从 Parquet 到 batch：nanochat 的输入到底是什么

### 6.1.1 数据形态：Parquet 里存的是文本，不是 token

nanochat 的 base/pretrain 数据集是一组 Parquet 文件（shards）。每个 Parquet 的 row group 里有一列名为 `text` 的字符串列表。DataLoader 直接读字符串，然后在加载过程中调用 tokenizer **在线分词**，而不是提前把 token id 数组落盘。

这会带来两个非常实际的工程结论：

1. **训练吞吐可能受分词速度影响**：尤其是 CPU 较弱或 buffer 太小的时候。
2. **tokenizer 与训练强绑定**：你换 tokenizer 不只是换词表，还会改变整个训练的序列统计特性（长度分布、BOS 的位置、特殊 token 的 bytes 计数等）。

### 6.1.2 “+1” 的小细节：为什么一行要装下 `T+1` 个 token

自回归语言模型的训练是“用当前 token 预测下一个 token”。因此，如果你希望模型一次前向/反向训练 `T` 个预测位置，你需要一段长度为 `T+1` 的 token 序列：

- 输入（inputs）：前 `T` 个 token
- 标签（targets）：后 `T` 个 token（整体右移一位）

也就是：

$$x = \text{row}[0:T],\quad y = \text{row}[1:T+1]$$

nanochat 的 pretrain DataLoader 正是用这个约定构造 `(B, T)` 的 inputs/targets（其中行缓存的容量是 `T+1`）。

### 6.1.3 边界约定：每篇文档都 prepend 一个 BOS（但不追加 EOS）

`nanochat/dataloader.py` 的核心约定是：**每个文档（Parquet 的每一行文本）在分词后都会在最前面显式插入一个 BOS token**。这由 `tokenizer.get_bos_token_id()` 给出。

注意这里的“BOS”是一个**起始分隔符**的工程语义：它告诉模型“这里是一个新文档的开始”。nanochat 的 base 预训练并不依赖一个明确的 “EOS（结束符）” 来划定文档结尾；文档结尾更多由因果注意力的方向性与 BOS 的重置信号共同承担。

**Rule-of-thumb：** 如果你来自那些“必须追加 EOS + block-diagonal attention mask”的实现，要特别留意：nanochat 的 base packing 不构造跨文档隔离的注意力掩码（见 6.2），它依赖 BOS 作为“软边界”。

---

## 6.2 BOS-aligned best-fit cropping：100% 利用率与“丢 token”的取舍

真实文本长度是长尾分布：短则几十 token，长则上万 token。你若采用“每篇文档单独截断/填充到 `T`”的朴素策略，会导致大量 padding token 参与前向/反向计算，吞吐直接被浪费。

nanochat 的 base DataLoader 采用一种很激进但很清晰的策略：**best-fit packing + cropping**，目标是做到：

- 每一行都装满（100% 利用率）
- 每一段文档都从 BOS 开始（BOS-aligned）
- 当必须裁剪时，尽量让“浪费最小”

### 6.2.1 算法直觉：先塞“最合身的整段”，再用裁剪把剩余填满

DataLoader 会维护一个文档 token 列表的缓冲区（`doc_buffer`）。构造每一行时重复执行：

1. 在缓冲里找**能完全塞进剩余空间**的文档里“最长”的那一个（best-fit：尽量减少碎片）。
2. 如果找不到任何文档能完整塞进剩余空间，就从缓冲里找**最短**的文档，把它从开头裁剪一段，正好填满剩余空间（这样丢掉的 token 最少）。

可以用一个小例子感受它的行为（假设一行容量 `T+1=11`，括号里表示 token 数）：

```text
------------------ 行容量 = 11 ------------------+
缓冲里的文档长度：A(6), B(5), C(4), D(3)

起始剩余 11：
  最长能塞进的是 A(6) -> 放入 A，剩余 5
剩余 5：
  最长能塞进的是 B(5) -> 放入 B，剩余 0
=> 本行无裁剪，0 padding，100% 利用率

若缓冲是：A(6), C(4), D(3)
起始剩余 11：
  放 A(6)，剩余 5
剩余 5：
  最长能塞进的是 C(4)，放 C，剩余 1
剩余 1：
  没有文档长度 <= 1，则取最短 D(3)，裁剪 D 的前 1 个 token 填满
=> 丢掉 D 的剩余 2 个 token（以及之后它本应出现的上下文）
```

这种策略非常“硬核”：它宁愿丢掉一部分 token，也要保证每一次 forward/backward 都在做有效训练，而不是在 padding 上烧钱。`nanochat/dataloader.py` 的注释给了一个经验数字：在 `T=2048` 时，大约有 ~35% 的 token 会因为裁剪而被丢弃（这是设计取舍的一部分，而不是 bug）。

实现上还有几个“只看代码才会注意到”的点，它们决定了你遇到瓶颈时该优先动哪些旋钮：

- **缓冲 refill 的单位是“文档批”**：DataLoader 内部先按 Parquet row group 读出一批字符串文档，再按 `tokenizer_batch_size` 切块 yield。`refill_buffer()` 拿到这一批字符串后，会调用 `tokenizer.encode(..., num_threads=tokenizer_threads)`，为每篇文档 prepend BOS，再把 token 列表逐个塞进 `doc_buffer`。因此，`tokenizer_threads`、`tokenizer_batch_size`、`buffer_size` 三者共同决定 CPU 分词吞吐与“best-fit 的可选空间”。
- **输出张量构造尽量“少拷贝、少分配”**：DataLoader 会预分配 CPU 上的 `row_buffer(B, T+1)`，再用一块连续的 pinned CPU buffer 做 staging，把 inputs/targets 拼在一起一次性拷到 GPU buffer，最后只用 view 切出 `(B,T)` 的 inputs/targets。这样把 HtoD 变成一次大拷贝，而不是很多次小拷贝，吞吐和稳定性都会好得多。

可以把它理解成下面这条简化的数据通路：

```text
Parquet(row group) -> text list -> tokenizer.encode(并行) -> doc_buffer
   -> best-fit/crop 填 row_buffer(B, T+1)
   -> pinned CPU staging (inputs|targets)
   -> 单次 HtoD copy -> GPU buffer -> views: inputs(B,T), targets(B,T)
```

### 6.2.2 关键差异：nanochat 的 packing 不做 block-diagonal attention mask

很多“packing 方案”会构造块对角因果掩码，让每个子文档互不可见；这样数学上最干净，但需要你在注意力内核里引入额外的分段信息（或特制 kernel）。

nanochat 的 base packing 选择了更极简的路线：**不构造额外 attention mask**，而是把文档片段按 token 序列直接拼起来，并确保每个片段都以 BOS 开头。这样它可以直接复用标准的因果注意力实现（FA3 或 SDPA），工程复杂度最低。

这也意味着：理论上，后面的文档 token 可以“看到”前面文档 token（因果方向允许）。nanochat 的设计希望 BOS 的强边界信号足以让模型学会“遇到 BOS 就忽略更早文本”，从统计上减少跨文档污染。

**Rule-of-thumb：** 如果你在验证集上看到生成出现明显的跨文档串味，优先检查 BOS 是否真的被正确 prepend、以及 tokenizer 的 BOS id 是否一致；不要先去怀疑模型架构。

---

## 6.3 DDP 分片：按 Parquet row group 分工，避免重叠

分布式数据并行（DDP）的基本要求只有一个：**同一步里，不同 rank 不能吃到重叠数据**。nanochat 的 base DataLoader 用 Parquet 的 row group 作为天然的分片单位：

- 对于一个 Parquet 文件，假设它有 `G` 个 row group；
- world size 为 `W`；
- rank 为 `R`；

那么该 rank 读取的 row group 索引序列是：

$$R,\ R+W,\ R+2W,\ \dots$$

也就是按 `step=W` 进行 stride 读取。这种分片方式非常简单、确定性强，而且不需要额外的 sampler/epoch shuffle 机制。

### 6.3.1 训练/验证 split：最后一个 Parquet 作为 val

nanochat 的一个硬约定是：**最后一个 Parquet 文件作为验证集，其余作为训练集**。这意味着：

- `split="train"`：使用 `parquet_paths[:-1]`
- `split="val"`：使用 `parquet_paths[-1:]`

这条约定非常适合“不断下载新 shard、训练集滚动增长”的场景，但也带来一个常见脚枪：

**Gotcha（非常常见）：** 如果你本地只有 1 个 Parquet 文件，那么训练集会变成空集合（因为 `[:-1]` 为空），DataLoader 会表现得像“卡住/空转”。解决办法是：至少下载 2 个 shard，或者显式修改 split 规则。

还有一个小而关键的安全细节：`nanochat/dataset.py` 下载 shard 时会先写到 `.tmp` 再原子重命名；而 `list_parquet_files()` 会过滤掉以 `.tmp` 结尾的文件。因此，即便下载中断，你也不会读到“半个 Parquet”；但相应地，如果目录里只有 `.tmp`，DataLoader 会认为“没有可用数据”。

最后，`_document_batches()` 是一个无限迭代器：当所有 Parquet 文件都遍历完，它会把 `epoch` 计数加 1 并从头再来。这让预训练脚本可以用“步数/算力”而不是“epoch”来定义训练终点。

---

## 6.4 断点续训：`state_dict` 能恢复到哪里，不能恢复到哪里

预训练跑数天是常态，断电/掉卡/抢占也常态。nanochat 会把 DataLoader 的进度以 `state_dict` 的形式塞进 checkpoint meta 里（在 `scripts/base_train.py` 保存的 `meta_*.json` 里你能看到它）。

### 6.4.1 base DataLoader 的状态：`pq_idx` / `rg_idx` / `epoch`

`tokenizing_distributed_data_loader_with_state_bos_bestfit()` 每次 yield 时都会给出：

- `pq_idx`：当前 Parquet 文件索引
- `rg_idx`：当前 row group 索引
- `epoch`：数据集循环轮次（从 1 开始）

恢复时，`_document_batches()` 会用这个状态做一个很重要的“防重放”动作：如果你恢复到同一个 Parquet 文件，它会把 row group 起点推进一个 stride（本质上是 “从下一个 row group 开始”），避免恢复后重复吃同一批数据。

### 6.4.2 为什么这仍然不是 bit-level exact resume

尽管保存了 `pq_idx/rg_idx/epoch`，但 nanochat 的 base packing 还有两类内部状态没有被持久化：

1. best-fit packing 的缓冲区内容（`doc_buffer`）与其当前的碎片形态；
2. tokenizer 批处理与 refill 的边界（会影响“哪些文档先进入缓冲”）。

因此，nanochat 的恢复是**近似恢复**：它保证不从明显的旧位置重放，但不能保证“恢复后训练到同一 step 的权重 bitwise 完全一致”。这是一种典型的工程取舍：实现极简、恢复可靠、代价可接受。

**Rule-of-thumb：** 如果你的实验目标是“严格对比两个实现的 bitwise 等价”，不要依赖当前的 DataLoader resume；你需要额外持久化缓冲区、采样器与 RNG 状态，或者干脆把数据变成严格可寻址的 token 流并按 offset resume。

---

## 6.5 对照：SFT 的 best-fit pad packing（为什么 SFT 不愿意丢 token）

同样是“把变长样本塞进固定的 `(B,T)`”，SFT 阶段在 `scripts/chat_sft.py` 里用的是另一种 best-fit：**best-fit + padding（不裁剪）**。原因很现实：

- base 预训练数据量巨大，丢掉一部分 token（换取 100% 利用率）通常可以接受；
- SFT 数据往往更稀缺、更昂贵（人类对话/高质量合成/任务数据），**随便裁剪会直接损失监督信号**。

因此 SFT DataLoader 的核心取舍是：当剩余空间装不下任何完整对话时，不再裁剪对话，而是用 BOS token 把剩余空间 pad 满，并把这些 padding 位置的 targets 设为 `-1`（ignore_index）。

从“形状约定”角度，它与 base 很像：同样按 `T+1` 构造一行、inputs/targets 右移一位；但从“信息保真”角度，它与 base 相反：**宁愿浪费一点算力，也不丢监督 token**。

还有一个值得记住的实现细节：tokenizer 的 `render_conversation()` 会返回 `(ids, mask)`，其中 mask 设计用于标记“哪些 token 应该产生监督”（例如把 user 侧、工具输出侧标成 0）；但当前的 `chat_sft.py` 数据生成器只使用 `ids`，主要只对 padding 部分做 `-1` 掩码。这意味着：如果你希望做“只训练 assistant 侧 token”的经典 SFT，你需要在数据准备或 loss 里显式把这个 mask 转成 ignore_index（这部分更详细的讨论会放在第 8 章）。

**Rule-of-thumb：** 看到 best-fit 时先问一句：“这是 crop 还是 pad？”——它决定了你是在优化吞吐（base）还是在保护监督信号（SFT），两者的排障路径也完全不同。

---

## 本章小结

- nanochat 的 base 数据是 Parquet 里的文本列 `text`；在线分词，按 `T+1` 规则构造 inputs/targets。
- 每篇文档都会 prepend BOS；base 预训练不依赖显式 EOS 来划定文档结尾。
- `BOS-aligned best-fit cropping` 做到 100% token 利用率，但会丢掉一部分 token（`T=2048` 经验上约 35%）。
- packing 不构造 block-diagonal attention mask；它依赖 BOS 作为软边界，从而复用标准因果注意力内核（FA3/SDPA）。
- DDP 分片按 Parquet row group stride 读取；验证集约定为“最后一个 Parquet 文件”。
- 断点续训保存 `pq_idx/rg_idx/epoch`，属于近似恢复：不易重放，但不保证 bit-level 完全等价。

---

## 练习题

1. **（基础）为什么构造一行训练数据时需要 `T+1` 个 token，而模型 forward 的输入/标签却是形状 `(B, T)`？**
   - *Hint：自回归训练是“右移一位”的 next-token prediction。*
   <details>
   <summary>查看提示与答案</summary>
   因为你要在 `T` 个位置上预测“下一个 token”。如果行长度只有 `T`，你只能得到 `T-1` 个预测对（第一个 token 没有上文，最后一个 token 没有下一个）。用 `T+1` 的行可以形成 `x=row[0:T]` 与 `y=row[1:T+1]`，正好得到 `T` 个训练位置，且形状都为 `(B,T)`，便于张量化和高吞吐。
   </details>

2. **（基础）best-fit packing 为什么要“优先选能完整放入的最长文档”？**
   - *Hint：想象你在装箱：先放大件还是先放小件？碎片会怎样？*
   <details>
   <summary>查看提示与答案</summary>
   优先选择“能完整放入的最长文档”可以最大化利用剩余空间，减少碎片，从而降低触发裁剪（cropping）的频率。先放小件会导致剩余空间变得零碎，最后更容易出现“没有任何文档能完整放入”，从而不得不裁剪（丢 token）。
   </details>

3. **（基础）当剩余空间装不下任何完整文档时，为什么要裁剪“最短文档”，而不是裁剪最长文档？**
   - *Hint：被裁剪的 token 是哪一部分？裁剪谁的“浪费”更小？*
   <details>
   <summary>查看提示与答案</summary>
   nanochat 的裁剪是“取文档开头的一段填满剩余空间”，文档剩余部分会被丢弃。若裁剪最长文档，会丢掉更多 token；裁剪最短文档能让“被丢弃的尾巴”最短，从而在必须裁剪时把浪费降到最低。
   </details>

4. **（挑战）DDP row group 分片：若 `world_size=3`，某个 Parquet 文件有 10 个 row group（索引 0-9），分别列出 rank0/1/2 会读取哪些 row group。**
   - *Hint：每个 rank 从自己的编号开始，步长为 world_size。*
   <details>
   <summary>查看提示与答案</summary>
   - rank0：0, 3, 6, 9  
   - rank1：1, 4, 7  
   - rank2：2, 5, 8  
   这种 stride 读取保证同一个 Parquet 文件内不同 rank 不重叠。
   </details>

5. **（挑战）解释“最后一个 Parquet 作为 val”这一 split 约定的优点与风险。**
   - *Hint：优点：简单；风险：文件数太少或分布漂移。*
   <details>
   <summary>查看提示与答案</summary>
   优点是工程极简：无需额外的随机划分或采样器，训练/验证的边界清晰，也便于“逐步下载更多训练 shards”。风险是：如果本地只有 1 个 Parquet 文件，训练集会变为空；另外如果最后一个 shard 的分布与前面不同（例如来源不同/质量不同），验证指标可能被系统性偏移。
   </details>

6. **（挑战）为什么 nanochat 的 base packing 不做 block-diagonal attention mask，而是选择 “BOS 软边界”？这带来哪些好处与潜在代价？**
   - *Hint：好处在于内核复用与实现简洁；代价在于隔离不是数学严格的。*
   <details>
   <summary>查看提示与答案</summary>
   好处：不需要自定义注意力掩码/分段信息，可以直接复用标准因果注意力实现（FA3/SDPA），工程复杂度低、性能路径清晰。潜在代价：后续文档 token 在理论上能看到前面文档 token，隔离不是严格的；需要依赖 BOS 的统计信号让模型学会忽略跨文档上下文，极端情况下可能出现串味或分布偏移。
   </details>

7. **（挑战）为什么 DataLoader 要预先分配 pinned CPU buffer + 单次 HtoD 拷贝？如果改成“每行临时创建张量 + 多次小拷贝”，会发生什么？**
   - *Hint：PCIe 拷贝的固定开销与碎片化。*
   <details>
   <summary>查看提示与答案</summary>
   pinned memory 允许异步 DMA，提高 HtoD 拷贝效率；把 inputs/targets 放在一块连续 buffer 里一次性拷贝，可以显著减少小拷贝的固定开销与同步点。若改成频繁创建小张量并多次拷贝，会增加 Python/allocator 开销，打断 GPU pipeline，表现为 GPU utilization 锯齿、吞吐下降，甚至出现显存碎片化导致的间歇性 OOM。
   </details>

---

## 常见陷阱与错误 (Gotchas)

1. **只有 1 个 Parquet 文件，训练集为空**
   - **症状**：训练像“卡住/空转”，或者 DataLoader 一直取不到有效 batch。
   - **修复**：至少下载 2 个 shard（保证 train 至少有 1 个、val 有 1 个），或修改 split 规则（不要用 “最后一个文件做 val”）。

2. **裁剪比例过高导致“数据看得太少”**
   - **症状**：在数据量本来就不大的实验里，loss 下降变慢或波动变大；你以为是优化器问题，实际是 token 被丢掉太多。
   - **修复**：理解 best-fit cropping 的设计前提是“数据极多”；如果你在小数据/长文为主的场景，考虑改用 padding 型 packing（SFT 里就是这种思路），或降低 `max_seq_len`/增大 buffer，让整段文档更容易完整放入。

3. **BOS 插入不一致（或 BOS id 配错）**
   - **症状**：生成出现跨文档串味、评估 BPB 异常、或不同阶段（base vs chat）格式对不上。
   - **修复**：确认 `tokenizer.get_bos_token_id()` 返回的就是你训练 tokenizer 时约定的 BOS（nanochat 自训 tokenizer 通常是 `<|bos|>`；兼容外部 tokenizer 时可能是 `<|endoftext|>`）。

4. **分词成为瓶颈导致 GPU 等数据**
   - **症状**：GPU utilization 呈锯齿（算完一段就空转），吞吐远低于硬件上限。
   - **修复**：从工程上思考：提高分词并行度（threads/batch）、增大 buffer（减少频繁 refill）、或确保使用高效的 tokenizer 实现；先用小模型验证瓶颈再上大规模跑。

5. **把“断点续训”当成 bit-level 精确恢复**
   - **症状**：你期望两次 run（中断+恢复 vs 一口气跑完）得到完全相同的权重与曲线，结果对不上。
   - **修复**：理解当前实现是近似恢复：保存了 Parquet/row group 位置，但没有保存 packing 缓冲与 RNG。若你需要严格等价，必须额外持久化更多内部状态或改造数据寻址方式。
