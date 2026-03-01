# 第 12 章：排障与最佳实践（把问题拆成「token × 吞吐 × 数值」三条线）

## 12.1 开篇段落

做模型训练与数据准备，最痛的往往不是“跑不起来”，而是“跑得很顺、看起来很对、最后能力却很差”。这类问题的共同特征是：它们经常不会直接报错，而是以 **静默错误** 的形式潜伏在流水线里——某个 mask 偏移 1 个 token、某个特殊 token 协议在训练/推理两端不一致、某个推理端强制注入的 token 被你当成了监督目标……最后表现出来就是：Loss 很平滑、BPB 也不难看，但生成像失忆、RL 学不会、Web 聊天“总感觉怪”。

nanochat 的工程哲学是“最小可读、最小可改”，因此排障也应该遵循同一个原则：**把问题拆解到最小可验证的事实**。本章提供一套在 nanochat 里高命中率的排障框架：把任何异常都先归类到三条线之一（或它们的交叉处）：

- **token 线**：你喂进去的 token 序列、你监督的 targets、你统计的指标，三者在“边界 token / ignore_index / packing”上是否严格一致？
- **吞吐线**：瓶颈是在数据侧（I/O/分词/搬运）还是算子侧（attention/compile/通信）？显存爆炸来自训练激活还是推理 KV cache？
- **数值线**：精度策略、优化器/调度、FP8/compile 的组合是否引入了“看似收敛但悄悄变差”的数值路径？

你会看到：nanochat 的很多“有意思的小细节”（例如 Engine 的 `token_masks`、Web SSE 的 `�` 防抖、FA3/SDPA 自动切换、`_orig_mod.` 前缀修复、eval bundle 的 file lock）本质上都是为了让这三条线更可控、更可解释。

---

## 12.2 三线定位法：先把问题归类，再动手

下面这个“症状 → 归类 → 最小检查”的流程，建议你背下来（至少背到能在脑中快速跑一遍）：

```text
症状出现了什么？
├─ A. 直接报错 / 直接炸（OOM、dtype mismatch、找不到 kernel）
│   ├─ 看报错栈顶：发生在 dataloader/engine/flash_attention/checkpoint 还是模型 forward？
│   └─ 先把变量降到最小：d12、小 batch、短序列、单机单卡
├─ B. 不报错但很慢（GPU 利用率低/波动、MFU 低）
│   ├─ 吞吐线：CPU/I/O/分词/搬运 还是 attention/通信/compile？
│   └─ 用最小基线对比：同一模型同一 batch，禁用/启用 FA3/FP8/compile
└─ C. 训练看着正常但能力很差（静默错误）
    ├─ token 线：抽样 decode + 对齐 targets/mask（特别是 ignore_index=-1）
    ├─ 协议线：特殊 token 是否一致（<|user_start|>…<|assistant_end|> 等）
    └─ 评测线：BPB/CORE/ChatCORE 的统计口径是否被特殊 token / ignore 掩码污染
```

**Rule-of-thumb：** 任何“静默错误”都不应该靠调温度、改学习率来碰运气；它应该先被你用“可验证的 token 事实”定位出来。

---

## 12.3 Token 线：静默错误最常见的来源

### 12.3.1 `ignore_index=-1` 是 nanochat 的硬约定（全链路都要对齐）

在 nanochat 里，忽略监督目标（不计入 loss/评测）的约定是 **`-1`**：

- 模型训练 loss：`nanochat/gpt.py` 的交叉熵 `ignore_index=-1`。
- SFT padding：`scripts/chat_sft.py` 把 padding 区间的 targets 设为 `-1`。
- RL：`scripts/chat_rl.py` 会用 Engine 的 mask 把 prompt/forced token 对应 targets 设为 `-1`。
- BPB 评测：`nanochat/loss_eval.py` 会显式避开 `<0` 的 targets（避免负索引污染 `token_bytes`）。

这条约定的价值在于：它把“哪些 token 参与监督/指标”变成了一个跨阶段一致的事实。但它也带来一个非常典型的静默错误：**你把本应有监督的 token 也设成了 -1**，于是 Loss 可能异常偏低、甚至几乎不动，模型看起来“很快收敛”，实际却没学到你以为它学的东西。

**最小检查清单：**

1. 从 DataLoader 抽一个 batch（训练/验证都抽），统计 `targets >= 0` 的占比。
2. 抽几条样本把 `inputs/targets` 反解码成文本，肉眼对齐“哪里开始算 loss”。
3. 确认你要监督的边界 token（例如 `<|assistant_end|>`）没有被误 mask 掉。

### 12.3.2 特殊 token 协议：别在“训练端一套、推理端一套”

nanochat 的对话与工具控制 token 是显式的（见 `nanochat/tokenizer.py`），核心集合包括：

- 对话边界：`<|user_start|> <|user_end|> <|assistant_start|> <|assistant_end|>`
- 工具边界：`<|python_start|> <|python_end|> <|output_start|> <|output_end|>`
- 以及全局的 `<|bos|>`

排障时你要特别关注三件事：

1. **训练数据是否“闭合”**：每个 assistant 回复是否以 `<|assistant_end|>` 收尾？（Engine 也把它当停止条件之一）
2. **推理拼接是否一致**：Web/CLI 拼出来的 token 序列是否真的和 SFT/RL 里见过的模式一致？
3. **外部输入是否泄漏控制 token**：如果用户能直接输入 `<|assistant_end|>` 这种字符串，你可能会把“控制面”暴露给“数据面”，导致提前停止或状态机乱跳（详见第 4 章的 token 泄漏 gotcha）。

**Rule-of-thumb：** 对外部不可信输入，永远只做“普通文本编码”；所有 `<|...|>` 控制 token 都应由系统插入，而不是让数据决定。

### 12.3.3 Packing 的两种世界：Base 是「裁剪填满」，SFT 是「填充不丢」

很多排障混乱来自一个误解：以为“packing 就是一件事”。在 nanochat 里，Base 预训练与 Chat SFT 的 packing 策略是**刻意不同**的：

- **Base（`nanochat/dataloader.py`）**：BOS-aligned best-fit **cropping**  
  目标是 100% 利用率：不 padding，填满每一行；当没有文档能完整放下时，会裁剪一段文档填满剩余空间。代码注释里给了一个很直觉的量级：在 `T=2048` 时，约 **35% token 会因裁剪被丢弃**。
- **SFT（`scripts/chat_sft.py`）**：BOS-aligned best-fit **pad**  
  目标是“对话不被截断”：宁可 padding（用 BOS padding），也不裁剪对话 token；同时把 padding 区间的 targets 设为 `-1`，避免它们进入 loss。

这两个策略各自合理，但它们会影响你解读现象的方式：

- 你在 Base 里看到“文档末尾总是被截断”，这可能不是 bug，而是设计（cropping）。
- 你在 SFT 里看到“batch 里很多 BOS”，这可能是 padding，而不是数据真的全是 BOS。

**nanochat 的一个关键设计点**是：Base packing **不构造 block-diagonal 的 attention mask** 来隔离文档；它依赖 “每行都以 BOS 开始” 作为一种软边界，让每个 token 至少能回看 BOS，从而降低跨文档拼接带来的语义污染（这是一个取舍：更高利用率、更简单实现，但文档边界不是硬隔离）。

### 12.3.4 工具注入的监督边界：SFT 默认不使用 render mask

`nanochat/tokenizer.py` 的 `render_conversation()` 会返回两样东西：`ids` 和 `mask`（哪些位置算 loss）。在理论上，这能让你把工具输出（系统注入的 `<|output_start|>...<|output_end|>`）从监督里剔除。

但在当前实现里（`scripts/chat_sft.py`），SFT data generator 只拿了 `ids, _ = tokenizer.render_conversation(conversation)`，**mask 被忽略**；它实际只做了“padding targets = -1”。这有两个直接后果：

1. 如果你的对话里包含工具输出段，它可能会被当作监督目标的一部分（除非你额外处理）。
2. 你不能假设“render mask 在训练中生效”；它只是一种能力，是否启用取决于训练脚本。

与之对照，RL 训练里 `Engine.generate()` 会返回 `token_masks`（采样=1，forced=0），`scripts/chat_rl.py` 会用它把 forced token 对应的 targets 设为 `-1`。这套链路是“推理事实 → 训练对齐”的好例子：强制注入的 token 不会被误监督。

---

## 12.4 吞吐与显存：训练与推理是两套“爆炸模型”

### 12.4.1 训练显存：静态项 vs 动态项

训练时的显存，你可以先按“四大块”做心算：

$$M_{\text{train}} \approx M_{\text{weights}} + M_{\text{grads}} + M_{\text{optim}} + M_{\text{acts}}$$

其中前三项更多是“静态项”（与参数量强相关），最后的激活显存是“动态项”（与 batch、序列长度、checkpointing 等强相关）。nanochat 的很多旋钮（例如 `--device-batch-size`、`--total-batch-size`、`--sequence-len`、模型 `depth`）本质上都在改变这四项的权重。

**Rule-of-thumb：** OOM 排障先动最廉价的旋钮：每卡 batch → 序列长度 → 并发/编译策略 → 模型规模。别一上来改数据或改优化器。

### 12.4.2 推理显存：KV cache 的线性爆炸（`max_tokens` 是硬旋钮）

推理端的显存爆炸更多来自 KV cache，而不是激活。Engine 的实现会按 `len(prompt) + max_tokens` 预分配 decode cache（见 `nanochat/engine.py`），因此 `max_tokens` 不只是“最多生成多长”，更是“你为 KV cache 预留多少显存”的直接上限。

粗略量级仍然是：

$$M_{\text{KV}} \propto 2 \times n_{\text{layer}} \times B \times L_{\text{cache}} \times n_{\text{kv\_head}} \times d_{\text{head}}$$

其中 $B$ 是并发样本数（Web 的并发由 worker 数决定，离线采样/RL 的并发由 batch/num_samples 决定）。

### 12.4.3 Attention kernel：FA3 只在 Hopper(sm90)，其他走 SDPA fallback

`nanochat/flash_attention.py` 提供了一个统一接口：优先使用 FlashAttention 3（FA3），否则回退到 PyTorch SDPA。它的检测逻辑非常明确：

- 只有 CUDA 且 compute capability major==9（Hopper, sm90）才启用 FA3；
- Ada(sm89)、Blackwell(sm100)、MPS、CPU 都会走 SDPA fallback。

这在排障里会表现为：

- **吞吐差异很大**：你以为“开了 flash attention 就一定快”，但硬件不满足时实际在跑 SDPA。
- **行为仍应一致**：fallback 的目标是“语义一致”，不是“速度一致”。如果你看到精度差异，优先怀疑的是 dtype/数值路径，而不是注意力实现本身。

**最小检查清单：**

1. 明确机器是哪一代 GPU（尤其是是不是 Hopper）。
2. 确认当前 run 实际走的是 FA3 还是 SDPA（别只看配置，优先看运行日志/断言/行为）。

### 12.4.4 数据侧吞吐：在线分词 + Parquet row group + 一次性 HtoD

nanochat 的 base dataloader（`nanochat/dataloader.py`）有几个非常“工程味”的优化点，排障吞吐时很有用：

- **Parquet row group stride 做 DDP 分片**：每个 rank 以 `rg_idx = ddp_rank` 起步，步长 `ddp_world_size` 扫 row group；恢复训练时只保存近似位置（`pq_idx/rg_idx/epoch`），并会“前进 1 个分片”避免重复（因此 resume 不精确，见 12.6）。
- **预分配 pinned CPU staging + 单次 HtoD**：输入与 targets 会先写入一块 pinned CPU buffer，再用一次 non_blocking 拷贝到 GPU buffer，减少碎片化与拷贝开销。
- **best-fit cropping 的 buffer 搜索**：用 buffer 找“能放下的最大文档”，直到放不下再裁剪填满，尽量降低浪费。

当你看到 GPU 利用率锯齿、GPU 在等数据时，优先从这里下手：I/O（磁盘/网络）、pyarrow 读取、分词线程数、以及 pinned memory 是否生效。

---

## 12.5 数值线：从“炸掉”到“悄悄变差”

### 12.5.1 dtype 的隐含约束：rotary cache 必须是 BF16

`nanochat/gpt.py` 在 forward 里对 rotary embeddings cache 有一个硬断言：`cos/sin` 必须是 BF16。这意味着：

- 即使你在推理端做了某些 dtype 实验，也要确保模型里这部分缓存与预期一致；
- 否则你可能会遇到“看起来无关”的 dtype/assert 报错，或更隐蔽的数值路径差异。

### 12.5.2 logit softcap：一种“便宜但常有效”的稳定性补丁

同一个文件里还有一个很典型的稳定性细节：在计算 loss 前，会把 logits 切回真实词表大小、转为 FP32，并做一个 `tanh` 形状的 **logit softcap**（把 logits 平滑压到有限范围）。

你不需要把它神化，但要理解它的定位：它是一种“廉价的数值保险丝”。当你引入更激进的训练策略（更大 LR、更高温度、更激进的混合精度）时，softcap 往往能减少极端 logits 导致的梯度尖刺。

### 12.5.3 FP8：它能让你更快，也能让你更难排障

nanochat 的 base 训练脚本提供了 `--fp8` 选项（见 `scripts/base_train.py` + `nanochat/fp8.py`），但它带来的排障复杂度是真实存在的：

- FP8 只适合用在 Linear 的 GEMM 主干；某些维度过小/不满足硬件要求的 Linear 会被过滤跳过（脚本会统计“转换了多少、跳过了多少”）。
- 训练可以开 FP8，但评测通常会临时禁用 FP8 回到 BF16，以获得更稳定/可比的指标（脚本里有 `disable_fp8()` 的上下文管理器）。
- compile/eager 下的数值路径可能不同：即便 loss 看起来下降，最终生成质量也可能“悄悄变差”。这类问题的最佳排障手段通常是：**用同一数据同一超参，做一次 BF16 baseline 对照**。

**Rule-of-thumb：** FP8 排障不要“混着调”。先确定 BF16 baseline 正常，再逐个打开 FP8、compile、FA3 等性能开关。

---

## 12.6 复现与恢复：nanochat 的 resume 是近似的（你要知道它近似在哪里）

### 12.6.1 base dataloader 的 resume：保存的是 “pq/rg/epoch” 而不是精确 token 游标

Base dataloader 的 `_document_batches()` 会产出 `(pq_idx, rg_idx, epoch)` 作为近似位置。恢复时，如果仍在同一 parquet 文件上，会把 `rg_idx` 先按 world size 折算，再 **前进 1 个分片**，避免重复数据。这让“恢复后不重放”更重要，但也意味着：

- 恢复点不是精确的 token 边界；
- DDP world size 改变、parquet 重打包、shuffle 逻辑变化都会让恢复语义变得不可比。

如果你对“逐 token 的可复现”有强需求（例如做非常严格的消融实验），你需要额外保存 RNG 状态与更细粒度的数据游标；nanochat 当前选择的是工程上更稳健的近似恢复。

### 12.6.2 checkpoint 加载：`_orig_mod.` 前缀与缺失配置键

`nanochat/checkpoint_manager.py` 里有两个很实用的“排障补丁”：

1. **修复 torch.compile 的 `_orig_mod.` 前缀**：加载 state dict 时会把所有 key 的 `_orig_mod.` 前缀移除，避免你在 compile/非 compile 间切换时“怎么都 load 不上”。
2. **补齐缺失的 config 键**：通过 `_patch_missing_config_keys()` 与 `_patch_missing_keys()` 做向后兼容，让旧 checkpoint 也能被新代码加载。

排障 checkpoint 失败时，建议按顺序检查：

- 你加载的 tokenizer 词表大小是否与 checkpoint 的 `vocab_size` 一致（代码里有 assert）。
- 你是否跨了“compile 前缀/配置字段”不兼容的边界。
- 你是否在 CPU/MPS 上加载 BF16 checkpoint（加载逻辑会把 BF16 张量转成 float）。

### 12.6.3 eval bundle 下载：file lock 是为了避免多 rank 互相踩

CORE 评测 bundle 的下载路径使用了 `download_file_with_lock()`（`nanochat/common.py`），会在 `NANOCHAT_BASE_DIR` 下创建一个 `.lock` 文件，保证多进程/多 rank 同时启动时只有一个下载者，其余等待锁释放。

这类机制的排障关键点是：**base dir 必须一致**。如果你在不同 rank 上设置了不同的 `NANOCHAT_BASE_DIR`，锁就失效了，仍可能出现并发下载/解压互相踩的异常。

---

## 12.7 推理与服务侧：当“模型没问题但交互很怪”

### 12.7.1 Web 的“看似小”细节：`�` 防抖、参数校验与 role 限制

`scripts/chat_web.py` 里有几处很值得复用的工程细节：

- **SSE streaming 的 UTF-8 防抖**：累计 token 后整体 decode，并在字符串末尾不是 `�` 时才发送增量，避免多字节字符被拆开导致前端闪烁乱码。
- **输入限制是字符级的**：最多 500 条消息、单条 8000 字符、总 32000 字符；采样参数范围 `temperature∈[0,2]`、`top_k∈[0,200]`、`max_tokens∈[1,4096]`。
- **role 只支持 `user/assistant`**：如果你需要 system 指令，把它合并进第一条 user（与 tokenizer 的 system 合并策略一致）。

排障时不要只看“前端表现”，要回到 token 事实：Web 最终拼出来的就是 `<|user_start|>...<|assistant_start|>` 这一串 token；你只要能把这串 token 拿出来，就能用 Engine/离线脚本复现问题。

### 12.7.2 Engine 的 `token_masks` 与 forced token：别把系统注入当成模型能力

工具调用时 Engine 会强制注入 `<|output_start|>...<|output_end|>`，并用 mask=0 标记（`nanochat/engine.py`）。如果你在训练/评测里忽略了这个事实，最常见的后果是：

- 你把 forced token 也算进了 loss / 指标，得到一个“虚假的学习信号”；
- 或者你让模型去拟合“环境返回值的字面形式”，反而把模型带偏。

**Rule-of-thumb：** 任何涉及工具的训练/评测，都应显式处理 forced token 的 mask（要么来自 render mask，要么来自 Engine 的 `token_masks`）。

---

## 本章小结

- 把排障拆成三条线：**token 对齐**（协议/targets/ignore_index）、**吞吐与显存**（训练激活 vs 推理 KV cache）、**数值路径**（dtype/FP8/compile/稳定性补丁）。
- nanochat 的 ignore_index 统一为 `-1`，并贯穿训练、SFT、RL 与 BPB 评测；静默错误最常见的形态就是“把该监督的 token 也 mask 掉了”。
- Base 与 SFT 的 packing 策略不同：Base 为 100% 利用率做 best-fit cropping（可能丢 token），SFT 为不丢对话做 best-fit pad（padding targets 置 -1）。
- 注意力实现会随硬件切换：FA3 只在 Hopper(sm90)，其余回退 SDPA；速度差异巨大但目标是语义一致。
- 复现与恢复要理解“近似在哪里”：base dataloader 保存的是 `pq/rg/epoch` 的近似游标；checkpoint loader 会修复 `_orig_mod.` 前缀并补齐缺失配置。

---

## 练习题

1. **（基础）你在 SFT 训练中看到 Loss 从第一步起就异常偏低且几乎不动。结合 nanochat 的 `ignore_index=-1` 约定，最可能是哪类静默错误？你会用哪三个“最小检查”来确认？**
   - *Hint：先别想学习率；先想有多少 targets 真正参与了 loss。*
   <details>
   <summary>查看提示与答案</summary>
   最可能是 targets 被大面积（甚至全量）设成了 -1，导致有效监督极少或为零。最小检查可以是：①统计 `targets>=0` 的占比（按 batch、按样本都看）；②抽几条样本把 inputs/targets 反解码对齐，确认 assistant 回复段确实在算 loss；③检查 padding/mask 的边界计算是否把 `<|assistant_end|>` 或整段回复误 mask 掉。只有先证伪“监督是否存在”，才有资格讨论超参收敛。
   </details>

2. **（基础）为什么 nanochat 的 Base packing 不做 block-diagonal attention mask 仍然“能用”？这种设计的收益与代价分别是什么？**
   - *Hint：BOS-aligned、cropping、以及“软边界”。*
   <details>
   <summary>查看提示与答案</summary>
   因为 Base dataloader 强制每行以 BOS 开始，让每个 token 至少能回看 BOS，从而在经验上降低跨文档拼接带来的语义污染；同时 best-fit cropping 追求 100% 利用率，不引入 padding 和复杂的 mask 构造。收益是实现简单、吞吐高、利用率高；代价是文档边界不是硬隔离，拼接处仍可能存在“跨文档上下文”的软污染，并且为了填满会裁剪丢弃一部分 token（注释里给出约 35% 的量级）。
   </details>

3. **（挑战）推理端 OOM 只在 Web 服务上出现、离线单条生成不出现。结合 KV cache 的线性公式，你会优先怀疑哪两个旋钮？为什么它们在 Web 场景下更敏感？**
   - *Hint：并发与 `max_tokens`。*
   <details>
   <summary>查看提示与答案</summary>
   优先怀疑并发（worker 数/同时请求数）与 `max_tokens`。KV cache 近似与 `B * (len(prompt)+max_tokens)` 成正比：Web 服务为了并发会同时跑多条生成（B 上升），并且可能允许较大的 `max_tokens`（为“能聊很长”预留），两者叠加非常容易触发 OOM；而离线单条生成通常是 B=1、且你更容易手动控制 `max_tokens`，因此不容易炸。
   </details>

4. **（挑战）你在非 Hopper GPU 上以为自己“启用了 FlashAttention 3”，但吞吐很差。结合 `nanochat/flash_attention.py` 的选择逻辑，最可能的真实情况是什么？你会如何验证？**
   - *Hint：compute capability major。*
   <details>
   <summary>查看提示与答案</summary>
   最可能的真实情况是：硬件不是 Hopper(sm90)，因此 FA3 不会被启用，实际在走 PyTorch SDPA fallback（语义一致但速度可能差很多）。验证思路是先确认 GPU 代际/compute capability（major 是否为 9），再结合运行日志/断言或在最小脚本里打印 `HAS_FA3`/实际走的实现路径，避免“配置写了就以为生效”的错觉。
   </details>

5. **（挑战）工具调用场景下，为什么“forced token 必须从 loss 里剔除”？如果不剔除，最可能出现哪两类偏差？**
   - *Hint：系统注入 vs 模型采样。*
   <details>
   <summary>查看提示与答案</summary>
   forced token（例如 `<|output_start|>...<|output_end|>`）不是模型采样出来的，而是系统根据外部执行结果强制写入上下文。如果不剔除：①你会把环境返回值当成“模型应该预测的金标准”，模型会被迫拟合一个它并未生成的序列，梯度方向是错的；②指标/评测会被污染：模型看起来“预测对了工具输出”，但那其实是系统注入，从而高估了模型能力或让 RL 的 reward/advantage 归因错误。
   </details>

6. **（挑战）解释 base dataloader 的 resume 为什么被称为“近似恢复”。在你做严格消融实验时，这会带来什么风险？你会怎么降低风险？**
   - *Hint：保存的是 pq/rg/epoch，不是逐 token 游标；world size 变化会破坏语义。*
   <details>
   <summary>查看提示与答案</summary>
   因为它只记录 parquet 文件索引、row group 索引与 epoch，并且恢复时会“前进一个分片”避免重复；这不是逐 token 的精确游标，也不保证在不同 world size/不同打包策略下语义一致。严格消融实验里，这可能导致恢复前后看到的数据分布发生不可控偏移，从而让对比不再纯粹。降低风险的方法包括：固定 world size 与数据版本；尽量避免中途改 packing/分词策略；对关键对比实验尽量从头跑到同一 step；或自行扩展保存更细粒度的游标与 RNG 状态。
   </details>

7. **（挑战）你加载 checkpoint 时遇到“key 不匹配/缺字段”，但你确信权重文件没坏。结合 `checkpoint_manager.py` 的设计，最可能的两个原因是什么？你会按什么顺序排查？**
   - *Hint：`_orig_mod.` 前缀与缺失 config 键。*
   <details>
   <summary>查看提示与答案</summary>
   最可能原因一是 torch.compile 引入了 `_orig_mod.` 前缀，导致 state dict key 与未编译模型不一致；原因二是代码版本演进导致 checkpoint 的 `model_config` 缺少新字段或权重缺少新模块的参数。排查顺序可以是：①确认 loader 是否做了 `_orig_mod.` 前缀移除；②检查 `meta_*.json` 里的 `model_config` 是否被补齐并与当前代码兼容；③确认 tokenizer 的 vocab_size 与 checkpoint 配置一致（避免“其实是词表不一致”）；④再考虑权重文件是否确实损坏。
   </details>

---

## 常见陷阱与错误 (Gotchas)

1. **把 `-100` 当成 ignore_index**
   - **症状**：你照搬别的仓库的 mask 逻辑，SFT/RL/BPB 里出现“指标离谱/训练不动/评测崩”。
   - **修复**：nanochat 的 ignore_index 是 `-1`，并且多处显式依赖 “targets<0” 这件事。

2. **以为 Base packing 会做 block-diagonal attention mask 或追加 EOS**
   - **症状**：你按“硬隔离文档”的直觉去解释现象，结果越解释越矛盾。
   - **修复**：Base 是 BOS-aligned best-fit cropping，不构造跨文档隔离 mask；文档边界是软的、且会裁剪丢 token。

3. **在 SFT 里假设 render mask 生效**
   - **症状**：你在数据里加入工具输出段，但训练后模型表现像“被工具输出带偏”或 loss/指标解释不通。
   - **修复**：当前 `chat_sft.py` 只 mask padding（targets=-1），不使用 render mask；需要你显式把工具段剔除或改训练脚本。

4. **Web 端使用 `system` role / 超出参数范围**
   - **症状**：请求 400，被提示 role 或温度/top_k/max_tokens 不合法。
   - **修复**：Web 只支持 `user/assistant`；并严格校验范围（`temperature∈[0,2]`、`top_k∈[0,200]`、`max_tokens∈[1,4096]`）。

5. **以为“启用了 FA3”就一定在跑 FA3**
   - **症状**：吞吐比预期差很多，但你找不到原因。
   - **修复**：FA3 只在 Hopper(sm90)；非 Hopper 一律 SDPA fallback。先确认硬件代际，再谈优化。

6. **FP8/compile/FA3 一起开，出了问题不知道该怪谁**
   - **症状**：loss 看着还行但生成变差，或者性能与数值表现都不稳定。
   - **修复**：做 BF16 baseline；然后一次只打开一个性能开关，用对照实验定位罪魁祸首。

7. **把近似 resume 当成精确复现**
   - **症状**：恢复后曲线“对不上”，或消融对比没有统计意义。
   - **修复**：理解 base resume 的近似性；严格实验尽量从头跑或自行扩展保存更精细的状态。

8. **多 rank 下载 eval bundle 互相踩**
   - **症状**：偶发下载/解压失败，或不同进程看到不一致的 bundle 内容。
   - **修复**：确保所有进程使用同一个 `NANOCHAT_BASE_DIR`，让 file lock 真正生效。
