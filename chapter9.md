# 第 9 章：RL（GSM8K 上的采样、reward、优势归一化与稳定性）

SFT 把模型从“纯续写器”塑造成“按对话协议说话的助手”，但它的本质仍然是行为克隆：模型只是在模仿参考轨迹（teacher forcing）。一旦推理时走偏一步，它就会迅速累积错误（曝光偏差），尤其在数学推理、代码生成这类“对错极其明确、长链条依赖极强”的任务上，SFT 往往会卡在一个不上不下的水平。

nanochat 的 RL 阶段选择了非常克制的目标：不搞复杂的价值网络（critic），不搞 PPO ratio/clip，也不引入对参考模型的 KL 约束，而是在 GSM8K 上做一种“GRPO 命名但更像 REINFORCE”的极简策略梯度训练（见 `scripts/chat_rl.py` 的注释）。本章将把它按代码真实行为讲清楚：采样怎么做、reward 怎么算、优势如何归一化、哪些 token 会参与梯度、以及这种简化方案最常见的稳定性陷阱是什么。

---

## 9.1 nanochat 的 RL 到底做了什么（以及没做什么）

先把边界画清楚，避免把一套“教科书 RLHF”投射到这份代码上：

- **做了**：对同一个问题采样多条回答；用规则 reward 打分；用策略梯度更新模型；用 token 级别的有效长度做归一化；在分布式环境下做 pass@k 评测与定期保存 checkpoint。
- **没做**：没有单独训练 value network；没有 PPO 的旧策略比值与 clip；没有 trust region；没有 KL 正则到参考模型（脚本开头明确说“删除 trust region，没有 KL”）。

因此你应该把它理解成：**在一个给定的对话协议与引擎（Engine）之上，用“采样→打分→优势→更新”的闭环，推动模型在 GSM8K 上更常走到正确答案。**

---

## 9.2 从问题到更新：一条最小 RL 闭环

### 9.2.1 样本与 prompt：`render_for_completion()` 是核心接口

在 `tasks/gsm8k.py` 里，每条样本被包装成一个 conversation，其中最后一条 assistant 消息是参考解答（包含 `####` 答案标记，且可能包含工具轨迹 `<<expr=result>>` 被解析成 parts）。

RL 采样时不能把参考答案喂给模型。因此 `scripts/chat_rl.py` 会对每个 conversation 调用 tokenizer 的 `render_for_completion()`：

- 删除最后一条 assistant 消息
- 渲染剩余消息（包含 BOS 与 `<|user_start|>...<|user_end|>`）
- 在末尾追加 `<|assistant_start|>`，把模型置于“该轮 assistant 需要生成”的起点

这一步决定了 RL 的“输入分布”是否与 SFT/推理一致：你若自行拼接 prompt，很容易在空格/换行/边界 token 上产生字节级偏移，导致 reward 学不到东西。

### 9.2.2 采样（rollout）：同一个问题采 `K` 条回答

RL 的第一个大头是采样：对同一个 GSM8K 问题，生成 `num_samples=K` 条不同回答。nanochat 用 `Engine.generate_batch()` 做 batched generation，并且为了避免 OOM，会把 `K` 拆成多个小批次（每次最多 `device_batch_size` 条）。

采样的随机性来自两处：

- `temperature>0` 时的 multinomial 采样
- `top_k` 截断（只在概率最高的 K 个 token 内采样，`top_k=0` 表示不截断）

为了让训练过程尽量可复现，脚本对每个 `(step, example_idx, sampling_step)` 组合生成一个确定性 seed（取 int32 正半区），并把它传给 Engine。你做排障时，如果怀疑“采样在抖”，优先检查这条 seed 链路是否被你改坏了。

### 9.2.3 reward：GSM8K 的规则打分是“提取 `####` 后的数值”

`tasks/gsm8k.py` 的 `reward()` 直接复用 `evaluate()`：它会从模型输出文本里用正则提取 `####` 之后的数字（并去掉逗号），与参考答案进行字符串匹配：

- 匹配成功：reward = 1.0
- 否则：reward = 0.0

这带来一个非常现实的结论：**reward 极其稀疏，而且强依赖格式**。模型如果不会输出 `####`，即便推理过程看起来合理，也会被判 0 分。

### 9.2.4 哪些 token 参与梯度：Engine 的 `mask` 决定“可训练位置”

这是 nanochat RL 里最有意思、也最容易踩坑的细节之一：`Engine.generate_batch()` 除了返回生成的 token 序列，还会返回一个与序列等长的 `mask`：

- prompt token：mask=0
- 正常采样出来的 token：mask=1
- 被引擎强制注入的 token（例如工具输出 `<|output_start|>...<|output_end|>`）：mask=0

RL 脚本会把不同长度的序列 pad 到同一长度（用 `<|assistant_end|>` 作为 padding token id，同时把 padding 的 mask 置 0），再构造自回归的 inputs/targets，并对 targets 做关键一步：

> 只要对应位置的 `mask==0`，就把该位置的 target 置为 `-1`（ignore_index）。

这确保了两件事：

1. 模型不会在 RL 里“学习去预测 prompt”（否则会把学习容量浪费在复制输入上）。
2. 模型不会去拟合“工具注入的 token”（否则会把工具系统与模型本体的边界打乱）。

**Rule-of-thumb：** RL 里最危险的 bug 之一，就是 mask 没对齐或 shift 错一位——你会在不知不觉中训练模型去预测它不该预测的东西，reward 可能还会上升，但推理系统会变得越来越怪。

### 9.2.5 优势（advantage）：只做 “减均值”，不做除标准差

对同一个问题采样得到 `K` 个 reward 后，nanochat 计算优势非常简单：

$$A_i = R_i - \mu,\quad \mu=\frac{1}{K}\sum_{i=1}^{K}R_i$$

也就是说：只做中心化，不做 z-score（不除以标准差）。脚本注释称其更像 DAPO 风格的 token-level 归一化：把“组内相对好坏”作为信号，而不是把 reward 强行拉到单位方差。

这个选择会引出一个你必须提前接受的事实：

- 如果某个问题的 `K` 条采样 reward 全部相同（全 0 或全 1），那么该组的所有优势都是 0，**这组数据不会推动策略更新**。

这就是所谓的冷启动问题：如果 SFT 基础太差，RL 可能完全学不动。

### 9.2.6 更新（policy gradient）：用 token 级别有效长度做归一化

在实现上，模型 forward 会输出每个位置的 NLL（负对数似然）。脚本取负得到 log-prob，再乘以优势，形成一个 REINFORCE 风格的目标：

$$J(\theta)=\sum_{t} A \cdot \log \pi_\theta(y_t\mid x,y_{<t})$$

实现细节上，脚本并不是显式去取 logits 再算 log-prob，而是复用模型的交叉熵实现：`model(inputs, targets, loss_reduction='none')` 返回每个位置的 NLL（即 $\ell_t=-\log\pi_\theta$）。由于 `ignore_index=-1` 的位置损失为 0，把损失取负就得到了“只在有效 token 上定义”的 log-prob，再与优势相乘并求和。

训练时最关键的稳定性措施是：对目标做 token 级别的有效长度归一化。脚本会统计 `targets>=0` 的有效 token 数 `num_valid`，把目标除以 `num_valid`（以及拆分采样的 pass 数、每步处理的 example 数），避免“序列越长梯度越大”的隐性偏置。

另一个朴素但有效的稳定性手段是“控制步长”：RL 脚本会把优化器的学习率先乘上 `init_lr_frac`（默认 0.05），并在整个训练过程中把学习率线性衰减到 0。没有 KL 约束时，较小的步长能显著降低策略漂移与格式崩坏的风险。

---

## 9.3 工程与分布式细节：一个 RL step 具体在做什么

如果你要排查 OOM、吞吐、或“分布式莫名 hang”，一定要能在脑子里展开 `chat_rl.py` 的嵌套循环。它的基本单位不是“一个 batch”，而是“一个 step 里处理若干个 GSM8K 题目，每题采样 K 次”：

```text
for step in range(num_steps):
  (可选) 每 eval_every 步：跑一轮 pass@k 评测

  for example_step in range(examples_per_rank):   # 每个 rank 负责的题目数
    # 1) 取 1 个题目 -> render_for_completion -> prompt tokens
    # 2) 对同一题目采 K 次（分多次 batched sampling 防 OOM）
    # 3) 算 reward -> advantages
    # 4) pad 序列 + mask targets -> 得到 inputs/targets
    # 5) 再把 (B,T) 切成若干 forward/backward pass 防 OOM
    # 6) backward 累积梯度

  optimizer.step()  # 更新一次
  (可选) 每 save_every 步：保存 checkpoint（仅 master）
```

其中几个参数的“量级关系”决定了你的显存与吞吐：

1. **每 step 的总序列数**：$$\text{sequences/step}=\text{examples\_per\_step}\times \text{num\_samples}$$  
   这不是 tokens 数，但它直接影响采样总量与训练总量。
2. **DDP 的 rank 分工**：脚本要求 `examples_per_step` 必须能被 `world_size` 整除，否则就无法保证每个 rank 在同一 step 里处理相同数量的题目（会引发同步问题）。
3. **采样与训练的两次“分批”**：
   - 采样时，如果 `num_samples` 大于 `device_batch_size`，会拆成多个 sampling pass；
   - 训练时，如果拼起来的 `inputs_all` 行数超过 `device_batch_size`，又会再拆成多个 forward/backward pass。

**Rule-of-thumb：** RL 的 OOM 往往先出现在“采样阶段”（KV cache × 并发样本数），其次才是“训练阶段”。排查时先动 `num_samples`、`max_new_tokens`、`device_batch_size`，再考虑动模型规模。

### 9.3.1 一个实用的显存估算：KV cache 近似与哪些旋钮线性相关

在 nanochat 的 Engine 里，decode 阶段会为 KV cache 预分配大张量；无论底层是 FA3 还是 SDPA fallback，cache 的“量级”都近似线性随以下量增长：

- `num_layers`（层数）
- batch（本次并发生成的样本数，RL 里通常等于某次 sampling pass 的 `device_batch_size`）
- `kv_seq_len`（大约等于 prompt 长度 + `max_new_tokens`）
- `n_kv_head` 与 `head_dim`

用 BF16（2 bytes）粗略估算，KV cache 显存可以写成：

$$M_{\text{KV}} \approx 2 \times n_{\text{layer}} \times B \times L \times n_{\text{kv\_head}} \times d_{\text{head}} \times 2\ \text{bytes}$$

最前面的系数 2 来自 K 和 V 两套缓存。这个公式不需要非常精确（真实实现还有对齐与额外 buffer），但足够用来做“哪三个旋钮最危险”的优先级判断：通常就是 `device_batch_size`、`max_new_tokens`、以及模型深度/宽度。

---

## 9.4 评测与 checkpoint：pass@k 为什么能快速告诉你“有没有学到东西”

### 9.4.1 pass@k：同一题采 k 次，只要有一次做对就算过

`chat_rl.py` 会定期在 GSM8K test 集上评测 pass@k。它的逻辑是：

- 对每个问题采样 `k` 条解答；
- 如果前 `k` 条里 **任意一条** 被评测为正确，则该题在 pass@k 下算通过；
- 汇总所有题得到 pass@k（k=1..device_batch_size）。

这种指标非常适合 RL 训练的直觉：RL 采样的目标之一就是让“正确解的概率质量变大”，当你增加采样次数时，pass@k 应该上升；如果 pass@k 完全不动，常见原因是 reward 太稀疏（冷启动）或 mask/格式对齐出了问题。

实现上还有个小细节：脚本在评测 pass@k 时使用的是非零温度的采样（默认 `temperature=1.0`），因此 pass@k 更接近“在当前采样策略下，给你 k 次机会能否撞对一次”。如果你想测“确定性上界”（例如看模型在贪心解码下能做对多少），可以把温度设为 0。并且要记住：在 nanochat 的采样实现里，`top_k` 只在 `temperature>0` 时才会生效；温度为 0 时直接 argmax，top-k 参数不会影响结果。

### 9.4.2 checkpoint：只存模型，不存优化器（意味着你不该指望从 RL 中断处精确续跑）

RL 脚本会由 master rank 定期保存 checkpoint 到 `NANOCHAT_BASE_DIR/chatrl_checkpoints/` 下，以 `d{depth}` 作为模型 tag。保存时：

- 只保存 `model.state_dict()` 与少量 `model_config`；
- **不保存 optimizer state**（动量缓冲等）。

这是一种很现实的工程取舍：RL 通常跑得比 base 预训练短得多，更强调“从 SFT 起点快速推一段”，而不是长时间稳定续跑。因此你在做实验管理时应当把 RL 当成“可重跑的小段”，而不是“必须精确 resume 的长跑”。

---

## 9.5 两个“很 nanochat”的细节：用 `assistant_end` padding、用 Engine mask 切断工具边界

1. **用 `<|assistant_end|>` 作为 padding token id**：脚本把不同长度的采样序列 pad 到同一长度时，会用 `<|assistant_end|>` 的 id 做填充。这在语义上并不表示“模型真的输出了结束符”，因为 padding 的 mask 会被置 0，对应 targets 会被置为 `-1`，不会参与训练。它只是一个“方便且一定在词表里存在”的 padding id。
2. **Engine 的 forced token = mask 0**：Engine 在工具调用后会把工具输出 token 强制注入回上下文，并把这些注入 token 的 mask 置 0。RL 训练再用这个 mask 把 targets 置 `-1`，从而保持边界：模型学会“何时调用、如何解释”，但不被迫去拟合“工具执行结果本身”。

## 本章小结

- RL 起点是 SFT 模型：`chat_rl.py` 加载 `sft` checkpoint，以 `render_for_completion()` 构造 prompt。
- 采样用 `Engine.generate_batch()`：随机性来自 `temperature` 与 `top_k`；脚本用确定性 seed 让 rollout 更可复现。
- reward 是 GSM8K 的规则匹配：提取 `####` 后的数字，匹配则 1，否则 0；稀疏且强依赖格式。
- 哪些 token 参与训练由 Engine 的 `mask` 决定：prompt 与工具注入 token 都会被 mask 掉，对应 targets 置 `-1`（nanochat 的 ignore_index 约定）。
- 优势只做减均值（不除方差）；更新是 REINFORCE 风格，按有效 token 数做归一化；没有 KL/critic/PPO clip。

---

## 练习题

1. **（基础）为什么在 nanochat 的 RL 里，如果某个问题的 `K` 条采样 reward 全部相同，这一步几乎不会更新模型？**
   - *Hint：优势 $A_i=R_i-\mu$。*
   <details>
   <summary>查看提示与答案</summary>
   因为优势是对 reward 减均值：若所有 reward 相同，则 $\mu=R_i$，因此 $A_i=0$。策略梯度项本质是 $A\cdot \log\pi$，优势为 0 时，该组对更新的贡献为 0（忽略数值噪声）。这就是冷启动问题：没有 reward 差异就没有方向。
   </details>

2. **（基础）给定一组 reward 为 `[0, 0, 1, 1]`，计算均值与每个样本的优势。**
   - *Hint：均值是 0.5。*
   <details>
   <summary>查看提示与答案</summary>
   均值 $\mu=(0+0+1+1)/4=0.5$。优势分别为 `[-0.5, -0.5, +0.5, +0.5]`。这表示后两条采样相对更好，会被鼓励；前两条相对更差，会被抑制。
   </details>

3. **（基础）为什么 RL 训练时要把 prompt token 的 targets 置为 `-1`（ignore_index），而不是也参与 loss？**
   - *Hint：如果参与了，模型会在 RL 阶段学什么？*
   <details>
   <summary>查看提示与答案</summary>
   如果 prompt 参与 loss，模型会在 RL 阶段浪费大量梯度去“预测输入本身”（复制 prompt），这既不提升 reward，也会污染策略梯度信号，甚至让模型朝着“更像复读机”的方向移动。mask 掉 prompt 可以把更新集中在真正影响 reward 的生成部分。
   </details>

4. **（挑战）Engine 返回的 `mask` 里，工具注入 token（例如 `<|output_start|>...<|output_end|>`）为什么应该是 0？如果你把它当成 1 去训练，会有什么风险？**
   - *Hint：工具输出在推理时来自外部系统。*
   <details>
   <summary>查看提示与答案</summary>
   工具输出不是模型“决定”的，而是系统执行工具后注入的。如果把它当成可训练 token，模型会被鼓励去拟合工具输出的分布，进而在没有调用工具时也幻觉性输出工具结果，或在调用工具后重复生成一遍输出，破坏推理状态机的一致性。mask=0 能保持边界：模型学会“发起调用与解释结果”，而不是“猜工具结果”。
   </details>

5. **（挑战）为什么要按有效 token 数 `num_valid` 对策略梯度目标做归一化？如果不做，会出现什么系统性偏差？**
   - *Hint：长回答 vs 短回答。*
   <details>
   <summary>查看提示与答案</summary>
   如果不按有效 token 数归一化，长序列会自然产生更大的目标和梯度（因为求和项更多），即使它们的 reward 相同。模型会被隐性偏置去偏好某种长度分布（例如倾向于更短或更长，取决于优化器与 reward 的耦合）。用 `num_valid` 归一化可以让每步更新更接近“每个 token 的平均贡献”，减少长度作为隐藏变量对训练稳定性的影响。
   </details>

6. **（挑战）nanochat 的 RL 没有 KL 正则与参考模型。你认为这样做的潜在风险是什么？如果要加一个最小 KL 约束，你会把它加在目标函数的哪里？**
   - *Hint：想想策略漂移与语言能力退化。*
   <details>
   <summary>查看提示与答案</summary>
   没有 KL 约束时，策略可能为了 reward 在局部分布上走得很远（策略漂移），导致语言质量退化、格式崩坏或出现 reward hacking。最小 KL 约束通常加在目标里作为惩罚项：$J(\theta)=\sum_t A\log\pi_\theta - \\beta\\,\mathrm{KL}(\pi_\theta\\|\\pi_{ref})$；实现上需要一个冻结的 reference 模型或旧策略分布，用来计算每步或每 token 的 KL，再与优势项一起形成总目标。
   </details>

7. **（挑战）在 GSM8K 这种“格式强依赖”的 reward 下，你会如何降低冷启动失败的概率？请从 SFT 数据与 reward 设计两个方向给出建议。**
   - *Hint：让模型更容易产出非零 reward。*
   <details>
   <summary>查看提示与答案</summary>
   SFT 方向：确保 SFT 数据里包含足够多 GSM8K 风格的解题轨迹，并且强调最终必须输出 `####` 答案格式；让模型在 RL 开始时就有一定概率（哪怕很小）能输出可被提取的答案。Reward 方向：可以在早期采用更稠密的 shaping（例如只要出现 `####` 且后面是合法数字就给小正分；或对中间步骤的格式给予弱奖励），等模型学会格式后再切回严格的 0/1 正确性 reward。
   </details>

---

## 常见陷阱与错误 (Gotchas)

1. **冷启动：reward 全 0 导致学不动**
   - **症状**：训练日志里平均 reward 长期为 0，loss 更新很小或几乎不变。
   - **修复**：回到 SFT：先把格式（`####`）与基础正确率打上去；或临时引入 reward shaping，让模型更频繁地产生非零优势信号。

2. **mask/shift 对齐错误导致“训练目标跑偏”**
   - **症状**：reward 可能上升，但推理行为越来越怪（重复、工具输出混乱、甚至开始复读 prompt）。
   - **修复**：检查 targets 的右移与 mask 的右移是否一致：脚本是对 `targets` 的位置使用 `mask_ids[:, 1:]` 来掩码。任何 off-by-one 都会把梯度打到错误 token 上。

3. **`max_new_tokens` 太短导致 reward 低估**
   - **症状**：模型能写推理过程，但经常来不及输出 `####` 最终答案就被截断。
   - **修复**：提高生成长度预算（同时注意 KV cache 的显存成本），并在评测侧使用一致的 `max_new_tokens`。

4. **`num_samples` 太大导致 OOM 或吞吐崩溃**
   - **症状**：采样阶段直接 OOM，或 step 时间暴涨。
   - **修复**：把 `num_samples` 拆成多次 `device_batch_size` 的采样 pass（脚本已经这么做）；必要时降低 `device_batch_size`、`max_new_tokens` 或并发样本数。

5. **格式分布偏移：训练数据与推理引擎的工具协议不一致**
   - **症状**：模型在训练中看起来“会用工具”，但在推理引擎里工具状态机经常失灵。
   - **修复**：确保训练与推理使用同一套 token 协议（`<|python_start|>...<|python_end|>` 与 `<|output_start|>...<|output_end|>`），并保持空格/换行的字节级一致；调试时用 token 序列对齐而不是纯文本猜测。
