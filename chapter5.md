# 第 5 章：预训练配方（主旋钮：`depth`，其余尽量自动）

预训练阶段最消耗精力的，往往不是“把模型写出来”，而是把一次训练跑成**可比较、可复现、可扩展**的工程实验：你需要让不同规模的实验点在同一套假设下可对齐（数据量、batch、学习率、权重衰减、注意力窗口……），否则你看到的曲线差异很可能只是“某个旋钮顺手改了”的副作用。

nanochat 的 `scripts/base_train.py` 选择了一条更接近“配方工程”的路线：把“规模”主要压缩为一个旋钮 `--depth`，其余参数尽量由规则推导，或者给出在仓库内部自洽的默认值（你仍可以覆盖它们，但默认路线是一条能跑通、能复盘的黄金路径）。本章将按“从输入到训练闭环”的顺序拆开它：模型形状怎么由 `depth` 变出来、训练终点如何被定义、batch/LR/weight decay 为什么这样缩放，以及一些只看代码才能注意到的细节（FA3/FP8/`torch.compile`、logit softcap、vocab padding）。

---

## 5.1 为什么要“单旋钮”：让每个实验点都可比较

在传统 LLM 预训练里，你可以同时调 10 个关键超参；但当每个实验都要花掉大量 GPU 小时时，“超参自由度”会变成灾难：你很难复盘某次表现变差究竟是因为数据 horizon 不同、batch 不同、还是 schedule 形状不同。

`base_train` 的策略是把大量耦合关系固化为一个映射：给定 `depth`，其余关键量尽量按同一套规则推导，保证每个实验点都能放到同一条“缩放曲线”上对比：

```text
                [用户输入：--depth = L]
                         |
                         v
  +--------------------------------------------------+
  | (A) 模型形状：n_embd / n_head / window_pattern     |
  +--------------------------------------------------+
                         |
                         v
  +--------------------------------------------------+
  | (B) 训练终点：num_iterations / target_flops /      |
  |     tokens:scaling_params ratio                   |
  +--------------------------------------------------+
                         |
                         v
  +--------------------------------------------------+
  | (C) 优化配方：total_batch_size / (Muon+AdamW) LRs  |
  |     / weight_decay / warmup-warmdown              |
  +--------------------------------------------------+
```

**Rule-of-thumb：** 做 scaling sweep（例如 `d12 → d24 → d36`）时，宁愿少改旋钮，也不要“顺手”把一堆参数一并换掉。否则你得到的不是缩放定律曲线，而是一堆不可归因的数据点。

---

## 5.2 从 `depth` 到模型形状：`aspect_ratio`、`head_dim` 与 `window_pattern`

`base_train` 虽然以 `depth` 为主旋钮，但模型形状并不是“拍脑袋”：它由 `aspect_ratio`、`head_dim` 与对齐规则共同决定。

### 5.2.1 `n_embd` 的推导：先按比例，再按 `head_dim` 对齐

在 `scripts/base_train.py` 的 `build_model_meta()` 里，隐藏维度（`n_embd`）的规则可以概括为三步：

1. 理想宽度：$$d_{\text{base}} = L \times r$$ 其中 $L=\text{depth}$，$r=\text{aspect\_ratio}$（默认 64）。
2. 向上对齐到 `head_dim` 的整数倍：$$d_{\text{model}} = \left\lceil \frac{d_{\text{base}}}{d_{\text{head}}} \right\rceil \cdot d_{\text{head}}$$ 其中 $d_{\text{head}}=\text{head\_dim}$（默认 128）。
3. 头数：$$n_{\text{head}} = \frac{d_{\text{model}}}{d_{\text{head}}}$$ 并默认设置 `n_kv_head = n_head`。

这条规则的价值在于：**把 head_dim 变成一个硬约束**（而不是事后补救的整除检查）。例如：

- `depth=12`：$d_{\text{base}}=768$，本身是 128 的倍数，所以 $d_{\text{model}}=768$，$n_{\text{head}}=6$（每头 128 维）。
- `depth=13`：$d_{\text{base}}=832$，向上对齐到 $896$，所以 $n_{\text{head}}=7$。

### 5.2.2 `vocab_size` 来自 tokenizer（且内部会做 vocab padding）

`base_train` 在初始化模型前会先加载磁盘上的 tokenizer，并读取 `vocab_size`。这意味着预训练配方与 tokenizer **强绑定**：checkpoint 的 `meta_*.json` 会写入这个 `vocab_size`，加载模型时会进行一致性断言。

一个容易忽略的细节在 `nanochat/gpt.py`：模型会把 `vocab_size` 向上 padding 到 64 的倍数用于效率（embedding 与 lm_head 的权重矩阵会略大），但 forward 里会把 logits **切回真实的 `vocab_size`**。因此：

- 你看 `state_dict` 的参数量，可能会比“理论 vocab_size×d_model”略大；
- 但训练/评估的损失与采样只发生在真实词表上。

### 5.2.3 `window_pattern`：滑动窗口注意力的“层级模式串”

`GPTConfig.window_pattern` 是一个由 `S`/`L` 组成的字符串，按层循环平铺（tile）：

- `L`：全上下文（左窗口 = `sequence_len`）
- `S`：半上下文（左窗口 = `sequence_len // 2`）
- 最后一层强制为 `L`（保证最终层能看全上下文）

默认模式是 `SSSL`：多数层用半窗口节省注意力开销，周期性插入全窗口层做“全局整合”。这对吞吐很敏感，但也强依赖底层注意力实现。

`base_train` 启动时会打印 FlashAttention 3 是否可用：FA3 只在 Hopper（sm90）上启用；没有 FA3 时会回退到 PyTorch SDPA。此时若继续用 `window_pattern != L`，训练会走一条显式 mask 的慢路径，脚本会直接警告“GPU 利用率会很糟”。因此：

**Rule-of-thumb：** 非 Hopper 环境（或 FA3 不可用）上，优先用 `--window-pattern L` 跑通与对齐；确认 FA3 可用后，再用 `SSSL` 类模式串做吞吐优化。

---

## 5.3 “算力预算”在代码里怎么落地：params、tokens、FLOPs

### 5.3.1 参数计数是分组的（而且 embedding 与 lm_head 不共享）

很多框架会默认“输入 embedding 与输出 lm_head 权重共享”，但 nanochat 的 `nanochat/gpt.py` 明确采用 **untied weights**：`wte` 与 `lm_head` 是两套独立权重。除此之外，模型还包含一些很“nanochat”的额外参数：

- `value_embeds`：交替层的 value embedding（配合门控，类似 value residual 结构）；
- `resid_lambdas` 与 `x0_lambdas`：每层的可学习标量，用于残差缩放与回注入初始 embedding；
- rotary embeddings buffer：作为非持久 buffer（`persistent=False`）存在，不进 checkpoint。

`GPT.num_scaling_params()` 会把参数拆为：`wte`、`value_embeds`、`lm_head`、`transformer_matrices`、`scalars`、`total`。这不是“统计癖”，而是为了后续 scaling laws / compute horizon 的定义更清晰。

### 5.3.2 训练终点（三选一）：`--num-iterations` / `--target-flops` / `--target-param-data-ratio`

`base_train` 对训练步数（`num_iterations`）有明确优先级：

1. 你显式给 `--num-iterations`；
2. 否则你给 `--target-flops`（结合 `estimate_flops()` 与 batch 反推步数）；
3. 否则走最常用的 `--target-param-data-ratio`（默认 10.5）。

这里最容易误解的是 `target_param_data_ratio`。在 `base_train.py` 里它对应的是：

- 用 `scaling_params = transformer_matrices + lm_head`（注意：不是 total params）；
- 目标训练 tokens 为 $$D_{\text{target}} = r \cdot N_{\text{scaling}}$$ 其中 $r=\text{target\_param\_data\_ratio}$（默认 10.5）；
- 对应步数 $$\text{num\_iterations} = \left\lfloor \frac{D_{\text{target}}}{B_{\text{total(tokens)}}} \right\rfloor$$

脚本会在日志里打印 `Tokens : Scaling params ratio`，用来让你在实验复盘时明确：当前 run 的“数据:参数比”到底是多少。

**Rule-of-thumb：** 不要把这个 ratio 自动等同于 “Chinchilla 的 20:1”。它是 nanochat 内部定义的实验假设，并且作用于特定的参数子集；你应该用日志中打印出来的实际 ratio 来做对齐与比较。

---

## 5.4 Batch/LR/Weight Decay：把“经验”写成可复现的推导

### 5.4.1 `--total-batch-size` 的单位是 tokens（不是 sequences）

nanochat 把 “全局 batch size”定义为每次 optimizer step 看到的 **tokens 总数**。这减少了很多歧义：同样的 `total_batch_size`，无论你是 $B=32,T=2048$ 还是 $B=64,T=1024$，优化器更新时看到的 token 数都一致。

如果你不提供 `--total-batch-size`，脚本会用一套非常具体的规则去预测“近似最优 batch”：

- 参考点固定为 `d12`：$B_{\text{ref}} = 2^{19}=524{,}288$ tokens；
- 训练 token 参考 $D_{\text{ref}}$ 来自同一套 `target_param_data_ratio` 规则；
- 用 Power Lines 论文的经验式：$$B_{\text{opt}} \propto D^{0.383}$$
- 最后把预测 batch round 到最近的 2 的幂（为了效率与工程离散化）。

### 5.4.2 设备 batch 与梯度累加：用一条等式把 OOM 变成“可控”

训练里真正受显存约束的是 `--device-batch-size`（每卡每次 forward/backward 处理多少条序列）。`base_train` 用一条严格等式把它与 `total_batch_size` 串起来：

$$\text{grad\_accum} = \frac{B_{\text{total(tokens)}}}{(\text{device\_batch} \times T \times W)}$$

其中 $T=\text{max\_seq\_len}$，$W=\text{world\_size}$。脚本会 assert 这个值必须是整数。

**Rule-of-thumb：** OOM 时优先减 `--device-batch-size`，让 `grad_accum` 自动变大；不要轻易改 `--total-batch-size`，因为那会改变优化轨迹（梯度噪声尺度）。

### 5.4.3 优化器是 “Muon + AdamW” 的混合体（参数分组非常关键）

`GPT.setup_optimizer()` 会把参数拆成两大类：

1. **矩阵参数（Transformer blocks 里的线性权重）**：走 Muon（`kind='muon'`），并按 shape 分组堆叠优化；
2. **非矩阵参数（embedding/lm_head/value_embeds/scalars）**：走 AdamW（`kind='adamw'`），学习率各不相同。

一些只看代码才会注意到的细节：

- `lm_head`（unembedding）和 `wte`（embedding）使用不同基准学习率；
- `value_embeds` 跟 embedding 走一套 LR；
- `resid_lambdas` 的 LR 额外乘了 $0.01$（极保守）；
- `x0_lambdas` 用了不同的 Adam betas；
- AdamW 组的 weight decay 被设为 0（Muon 组才有 weight decay）。

除此之外，还有两层系统化缩放：

1. batch 变大时，学习率统一按 $$\eta \propto \sqrt{B/B_{\text{ref}}}$$ 缩放；
2. AdamW 组再按 $$\eta \propto 1/\sqrt{d_{\text{model}}}$$ 做 `dmodel_lr_scale`（以 768 为基准）。

Muon 还有两个“随训练推进而变化”的调度细节：

- **动量 warmup**：Muon 的 momentum 会在前约 300 步从 0.85 线性爬升到 0.95（更像一个“让优化器慢慢变得更惯性”的启动过程），避免一开始就用很大的动量把随机初始化的噪声放大。
- **权重衰减 warmdown**：Muon 的 weight decay 会在训练过程中线性衰减到 0（见下一小节）。这相当于把“强正则”更多放在早期，让后期更专注于细化收敛。

### 5.4.4 Weight Decay：先按理论缩放，再线性衰减到 0

`base_train` 先用一个“$T_{\text{epoch}}$”框架（把 $$T_{\text{epoch}}=\frac{B}{\eta\lambda D}$$ 维持常数）来缩放 weight decay：

$$\lambda = \lambda_{\text{ref}} \cdot \sqrt{\frac{B}{B_{\text{ref}}}} \cdot \frac{D_{\text{ref}}}{D}$$

然后在训练过程中再把 weight decay 从该值线性衰减到 0，保证“越到后期越少扰动”。

### 5.4.5 学习率 schedule：线性 warmup / constant / 线性 warmdown（不是余弦）

必须明确：nanochat 的 `base_train` 默认不是余弦调度，而是三段式线性形状：

- warmup：前 `warmup_ratio * num_iterations` 线性升到 1
- constant：中间保持 1
- warmdown：后 `warmdown_ratio * num_iterations` 线性降到 `final_lr_frac`

恢复训练时，只要 `step` 恢复正确，这个 schedule 天然连续，不需要额外保存“调度器状态文件”。

---

## 5.5 性能与数值：FA3、FP8、`torch.compile`、logit softcap

预训练配方里最“值钱”的工程细节，很多都来自下面四个点：

1. **FlashAttention 3 自动切换**：`nanochat/flash_attention.py` 会在 Hopper（sm90）上使用 FA3，在其他硬件回退到 SDPA。`base_train` 启动时会明确打印是否启用 FA3。
2. **FP8 训练（可选）**：`--fp8` 会把满足条件的大型 `nn.Linear` 替换成 Float8 版本（维度需能被 16 整除，且足够大）。评估时会临时把这些模块 swap 回 BF16/FP32 版本，避免评估指标被 FP8 数值误差污染。
3. **`torch.compile` 的双刃剑**：脚本会 `torch.compile(model, dynamic=False)` 提升吞吐，但保存 checkpoint 用的是 `orig_model.state_dict()`，避免编译 wrapper 引入的 key 前缀污染。
4. **logit softcap**：`nanochat/gpt.py` 在计算 logits 后会在 FP32 中做 $$z \leftarrow s \cdot \tanh(z/s)$$ 的 softcap（$s=15$），用于抑制极端 logit 带来的数值尖刺。

补充两个“只看实现才会注意到”的点：

- **FLOPs 估算会受 `window_pattern` 影响**：`GPT.estimate_flops()` 并不是只用 $6ND$ 的拍脑袋常数，它会把“矩阵参数的 6 FLOPs/param/token”与“注意力里 $QK^\top$ 的额外 FLOPs”分开估计；启用滑动窗口后，每层的有效注意力长度会被窗口截断，所以同一个 `depth` 在不同 `window_pattern` 下的 “FLOPs per token” 也会变。这也是为什么 `window_pattern` 在 nanochat 里既是质量旋钮，也是吞吐旋钮。
- **embedding/value_embeds 会在 CUDA 上转成 BF16**：`init_weights()` 里会把 `wte` 和 `value_embeds` cast 到 BF16 以节省显存（优化器仍能容忍这种精度），这能在小卡上多挤出一点 batch 空间。

---

## 本章小结

- `depth` 是主要规模旋钮；`aspect_ratio/head_dim/window_pattern` 决定形状与注意力窗口，但默认值构成一条“能跑通、能对齐”的黄金路径。
- `n_embd` 的推导是 “$L \times r$ 后向上对齐到 `head_dim` 的倍数”，从而保证 head_dim 可控且整除关系恒成立。
- 训练终点三选一：`--num-iterations` / `--target-flops` / `--target-param-data-ratio`；默认 ratio=10.5 且作用于 `transformer_matrices + lm_head`。
- batch size 会随 token horizon 按 $D^{0.383}$ 增长并 round 到 2 的幂；学习率按 $\sqrt{B/B_{\text{ref}}}$ 缩放；weight decay 先按理论缩放、再线性衰减到 0。
- 优化器是 Muon（矩阵）+ AdamW（embedding/lm_head/value_embeds/scalars）的混合体；参数分组与 LR 缩放是配方的核心之一。

---

## 练习题

1. **（基础）为什么 nanochat 更愿意让你输入 `depth`，而不是让你直接输入“目标参数量 N”？**
   - *Hint：同样的参数量可以对应不同的张量形状；硬件效率依赖整除与对齐。*
   <details>
   <summary>查看提示与答案</summary>
   因为参数量无法唯一确定 `n_embd/n_head/head_dim` 等形状参数，容易产生不整除、难并行、效率差或与仓库假设不兼容的配置。用 `depth` 作为主轴，再用固定规则推导形状，可以把大量工程约束（对齐、head_dim 约束）写死，从而让不同实验点可比较、可复盘。
   </details>

2. **（基础）给定 `depth=13`、`aspect_ratio=64`、`head_dim=128`，计算 `n_embd` 与 `n_head`。**
   - *Hint：先算 $d_{base}=L\times r$，再做向上对齐。*
   <details>
   <summary>查看提示与答案</summary>
   $d_{base}=13\times 64=832$。向上对齐到 128 的倍数得到 $d_{model}=896$。因此 $n_{head}=896/128=7$。
   </details>

3. **（基础）解释 `total_batch_size`（tokens）与梯度累加的关系。**
   - *Hint：用公式 $\text{grad\_accum} = B_{\text{total}}/(B_{\text{device}}\cdot T\cdot W)$。*
   <details>
   <summary>查看提示与答案</summary>
   `total_batch_size` 表示一次 optimizer step 看到的 tokens 总数。每张卡一次 forward/backward 消耗 `device_batch_size * max_seq_len` 个 token，全局乘上 `world_size`。为了让一次更新消耗的 tokens 精确等于 `total_batch_size`，需要用梯度累加把多个 micro-step 的梯度加起来，累加步数正是上述公式。
   </details>

4. **（挑战）为什么 `target_param_data_ratio` 用的是 `transformer_matrices + lm_head`，而不是 `total`？这会影响你如何解读 ratio？**
   - *Hint：embedding 是查表；不同参数子集对 scaling 拟合的“干净程度”不同。*
   <details>
   <summary>查看提示与答案</summary>
   这是为了让 scaling laws 假设更稳定：矩阵乘法参数与 FLOPs/损失缩放关系更直接；embedding 受词表大小强影响且更像查表；额外标量或 value embedding 的占比与作用也可能不符合传统假设。因此 nanochat 选择用 `transformer_matrices + lm_head` 做 scaling params。解读 ratio 时必须明确：它不是“总参数的 20:1”，而是“某个参数子集上的实验假设”，真正用于对齐比较的是日志打印出来的实际 ratio。
   </details>

5. **（挑战）`warmup_ratio=0`、`warmdown_ratio=0.5`、`final_lr_frac=0` 时，在训练进度 75% 处，`lr_multiplier` 约等于多少？**
   - *Hint：75% 已经进入 warmdown 且过了一半。*
   <details>
   <summary>查看提示与答案</summary>
   warmdown 占最后 50%。进度 75% 相当于 warmdown 进行到一半，线性从 1 降到 0，因此 `lr_multiplier≈0.5`。
   </details>

6. **（挑战）为什么脚本会把自动预测的 `total_batch_size` round 到最近的 2 的幂？这对吞吐与实验体系有什么好处？**
   - *Hint：硬件效率与实验点离散化。*
   <details>
   <summary>查看提示与答案</summary>
   很多 kernel/流水在对齐形状上更高效；同时把 batch 离散到少量可预期的值，可以减少“某个点 batch 只差一点点导致行为不同”的偶然性，让 sweep 更易管理与复盘。代价是 batch 可能略偏离理论最优，但换来工程稳定性与可比较性。
   </details>

7. **（挑战）FP8 转换为何要过滤 “维度能被 16 整除且足够大” 的 Linear？如果强行把小 Linear 也转 FP8，可能出现什么问题？**
   - *Hint：硬件约束 + 量化误差占比。*
   <details>
   <summary>查看提示与答案</summary>
   FP8 GEMM 往往要求特定维度对齐（常见为 16 的倍数）。另外小矩阵算力占比低、却更容易受量化误差影响，收益小、风险高（收敛变差或 kernel 不支持）。过滤能把 FP8 放在最能省时且最不容易翻车的地方。
   </details>

---

## 常见陷阱与错误 (Gotchas)

1. **把 `total_batch_size` 当成“序列条数”**
   - **症状**：你以为 `total_batch_size=524288` 是“52 万条样本”，于是修改 `max_seq_len` 后不理解为什么训练曲线变了。
   - **修复**：记住它的单位是 tokens。`max_seq_len` 变化会影响每次 micro-step 的 token 消耗，从而改变梯度累加步数与吞吐。

2. **把 `target_param_data_ratio` 误读成 “Chinchilla 固定 20:1”**
   - **症状**：你看到默认是 10.5 就以为配方错了，直接改成 20，结果与仓库内的基线（leaderboard/scaling 记录）对不上。
   - **修复**：先明确 ratio 作用于 `transformer_matrices + lm_head`，再用日志里打印的 `Tokens : Scaling params ratio` 来对齐实验假设。

3. **非 Hopper GPU 还开 `window_pattern=SSSL`，吞吐断崖式下跌**
   - **症状**：训练能跑但 MFU 很低，step 时间巨大。
   - **修复**：要么确保 FA3 可用（Hopper 环境），要么先用 `--window-pattern L` 跑通与对齐，再做滑窗模式的性能实验。

4. **恢复训练只加载权重，不加载 optimizer shard**
   - **症状**：resume 后 loss 抖动、收敛速度明显变慢。
   - **修复**：用 `--resume-from-step` 恢复，并确保每个 rank 的 `optim_..._rank{rank}.pt` 都存在且可读。学习率/weight decay schedule 由 `step` 计算，因此 `step` 必须与 checkpoint meta 对齐。

5. **tokenizer 与 checkpoint 的 `vocab_size` 不一致**
   - **症状**：加载时报 “Tokenizer vocab size does not match model config vocab size”。
   - **修复**：checkpoint meta 里记录了 `vocab_size` 并在加载时 assert。要么使用同一个 tokenizer 目录，要么重新训练 tokenizer 并从头训练模型，避免“半路换词表”。
