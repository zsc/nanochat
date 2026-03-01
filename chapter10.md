# 第 10 章：推理与交互（Engine/KV Cache、采样、工具、CLI/Web）

训练完一个 checkpoint，只意味着你得到了“一个能算 logits 的函数”。要把它变成“能聊的系统”，你还需要推理引擎：把 token 序列送进模型、把下一 token 从分布里采出来、维护 KV cache 降低解码成本、在合适的边界停止、以及（如果有工具）把模型输出与外部执行结果拼成一个一致的上下文。

nanochat 的推理链路有一个很强的工程特点：它尽量不引入庞大的第三方推理框架，而是把关键逻辑写成一套透明的最小实现（`nanochat/engine.py` + `scripts/chat_cli.py` + `scripts/chat_web.py`）。这对模型训练/数据准备人员的价值在于：你可以把“训练时看到的 token 分布”与“推理时喂入的 token 序列”逐项对齐，避免那类“只差一个空格就全盘崩掉”的分布偏移。

---

## 10.1 Engine：token in / token out（prefill + decode + KV cache）

### 10.1.1 Engine 的边界：它只理解 token id

`Engine` 的输入是“token id 列表”，输出是“下一 token id”。它不做文本模板渲染；对话边界 token（例如 `<|user_start|>`）由上层（CLI/Web/评测脚本）负责拼好。唯一的例外是：为了工具调用，它会持有 tokenizer（仅用于把工具表达式 token 解码成字符串、把工具结果再编码回 token）。

### 10.1.2 Prefill 与 Decode：为什么要“prefill 一次、再用 cache 解码”

自回归生成可以分成两段：

1. **Prefill**：把整个 prompt（长度通常很长）一次性送入模型，建立起 KV cache，并得到第一步的 logits。
2. **Decode**：每次只喂一个新 token，在 KV cache 上增量计算下一 token 的 logits。

如果每一步都把“全历史”重新前向一遍，你会得到 $O(T^2)$ 的总成本；KV cache 的意义就是把它降到“每步近似 $O(T)$ 的注意力访存 + 常数级的前向”，从而让长对话可用。

### 10.1.3 KVCache 的形状：为 FA3 设计的布局（与很多实现不同）

nanochat 的 `KVCache` 明确是为 FlashAttention 3 的 `flash_attn_with_kvcache` API 设计的。它的关键特征是：

- cache 的 head 维度布局是 **(B, T, H, D)**，而不是很多旧实现常见的 (B, H, T, D)
- cache 在推理时会被 in-place 更新
- 当前位置由 `cache_seqlens`（int32）追踪

在工程上，这会影响你做显存估算与并发控制的方式：并发样本数 `B` 与上下文长度 `T` 都会线性放大 KV cache 的显存。

### 10.1.4 一个很实用的优化：先 batch=1 prefill，再复制 cache

`Engine.generate()` 的一个“很 nanochat”的技巧是：

- 先对 prompt 做 **batch=1** 的 prefill（只算一次）
- 再把 KV cache 复制成 `num_samples` 份，让后续 decode 可以并行生成多个样本

这非常适合“同一 prompt 采样多条回答”（评测、RL、或 UI 的多样化采样），并且能避免重复计算 prompt 的前向。

### 10.1.5 停止条件：`<|assistant_end|>` 或 BOS

nanochat 的 Engine 在生成循环中把两类 token 视为“终止”：

- `<|assistant_end|>`：对话回复结束（SFT 会教模型学会输出它）
- BOS：如果模型生成了 BOS，也会被视为结束（作为一种兜底）

`Engine.generate_batch()` 返回的结果会**剔除终止 token**（终止 token 不会被加入最终序列），这对下游评测和 RL 的 padding/mask 处理很方便。

### 10.1.6 一个直接影响 OOM 的细节：decode cache 会按 `len(prompt)+max_tokens` 预分配

Engine 在 decode 阶段会创建一个“大 KV cache”，其长度 hint 大致是：

$$L_{\text{cache}} \approx L_{\text{prompt}} + L_{\text{gen}}$$

其中 $L_{\text{gen}}$ 在实现里通常来自 `max_tokens`。这意味着：`max_tokens` 不仅决定“最多生成多长”，也决定“你为 KV cache 预留多少显存”。即便模型很早就输出了 `<|assistant_end|>` 停止，cache 也已经分配完了。

因此在工程上，`max_tokens` 是推理 OOM 的一线旋钮（与并发样本数一起决定 KV cache 体积）。粗略估算你可以记住：

$$M_{\text{KV}} \propto 2 \times n_{\text{layer}} \times B \times L_{\text{cache}} \times n_{\text{kv\_head}} \times d_{\text{head}}$$

最前面的 2 来自 K 与 V 两套缓存。这个估算不要求精确，但足以用来解释“为什么并发一上来就炸”。

### 10.1.7 `token_masks`：把“模型采样”与“系统强制注入”区分开

nanochat 的 `Engine.generate()` 不只是逐步吐出 token id，它还会在同一列返回一个 `token_masks`：对 batch 内每一行给出 0/1 标记，表示该 token **是模型采样出来的（1）**，还是 **系统强制注入的（0）**。

这件事看起来像“额外的工程细节”，但它会在两个地方变成你排障与训练对齐的关键证据链：

- **工具调用**：当模型触发 `<|python_start|>...<|python_end|>` 后，`<|output_start|>...<|output_end|>` 是系统注入的，必须是 mask=0（否则你会把“环境返回的答案”错当成模型该预测的 token）。
- **RL/评测的 loss 对齐**：RL 里常常要把 prompt token（以及 forced token）从 loss 里剔除。`token_masks` 给你一个“推理侧事实”，能把这类对齐从“靠约定”变成“可验证”。

如果你用的是 `Engine.generate_batch()`（非流式、一次性返回最终序列），它也会同时返回 `masks`：prompt 部分通常被标记为 0，生成出来的部分再按采样/强制注入标记为 1/0，并且终止 token（`<|assistant_end|>` 或 BOS）会被剔除不出现在结果里。

### 10.1.8 一个容易踩的实现约束：KV cache dtype 的“仓库级假设”

`Engine.generate()` 里有一个直白但也很“硬”的实现假设：**CUDA 上 KV cache 用 BF16，非 CUDA 用 FP32**。它之所以这么写，是因为 KV cache 需要在生成前就预分配，而 `Engine` 本身并没有一套独立的“device/dtype 配置传递”机制。

这意味着：如果你改了推理端的 dtype 策略（例如在 CUDA 上强制用 FP32），或者你在同一仓库里引入了更复杂的混合精度/量化推理，**要记得同步 KV cache 的 dtype 选择**，否则会出现“能跑但慢/不稳”或“直接 dtype 报错”的非直观问题。

---

## 10.2 采样：只有 `temperature` 与 `top_k`（没有 top-p）

在 nanochat 的采样实现里，核心只有两个旋钮：

1. **温度（temperature）**：$T=0$ 时贪心解码；$T>0$ 时做随机采样。
2. **Top-k 截断（top_k）**：只在概率最高的 K 个 token 内采样；`top_k=0` 表示不截断、在全词表采样。

实现上非常直白：

- `temperature==0`：直接 argmax（确定性）
- `temperature>0 且 top_k>0`：取 top-k logits，softmax 后 multinomial
- `temperature>0 且 top_k<=0`：全词表 softmax 后 multinomial

**Rule-of-thumb：**

- **评测时**：设 `temperature=0`（确定性，上界测量）。
- **聊天时**：设 `temperature≈0.6~0.8`，`top_k≈50`（nanochat CLI/Web 默认就接近这套）。
- **写代码/数学推导时**：降低温度（例如 `0.0~0.2`），必要时降低 `top_k`，减少随机性。

---

## 10.3 工具调用：`<|python_start|>...<|python_end|>` + 强制注入 `<|output_start|>...<|output_end|>`

很多人把“工具调用”想成复杂的 JSON schema、函数签名、甚至多轮协议；nanochat 的工具调用更像一个非常明确的 token 级状态机：模型只需要学会在合适的时候输出一段被 `<|python_start|>` 与 `<|python_end|>` 包裹的表达式，剩下的由引擎接管。

### 10.3.1 引擎状态机：forced token 的关键语义

Engine 在每条样本（row）上维护一个 RowState：

- `in_python_block`：是否处于 python 段内
- `python_expr_tokens`：当前表达式的 token 缓冲
- `forced_tokens`：一个队列，表示“接下来必须注入”的 token（mask=0）

当模型生成 `<|python_end|>` 时，引擎会把表达式 token 解码成字符串，调用内部的 “calculator” 执行；若得到结果，则把：

`<|output_start|> + (结果的 token) + <|output_end|>`

压入 `forced_tokens` 队列。之后的若干步，引擎会优先从队列里吐出 forced token（即使模型当步采样出了别的 token，也会被覆盖），并在输出的 `token_masks` 里标记这些 forced token 的 mask=0。

这套 mask 对训练/评估很关键：RL 脚本正是用它把工具注入 token 从 loss 里剔除（见第 9 章）。

### 10.3.2 “python”其实是安全计算器：它能做什么、不能做什么

`use_calculator()` 的设计目标不是“执行任意 Python”，而是给 GSM8K 这类任务一个安全的算术工具：

- 允许纯算术表达式（并去掉数字中的逗号）
- 禁止 `**` 幂运算（以及一系列危险模式）
- 额外允许非常有限的字符串操作（目前只放开 `.count()`）
- 用 `signal.alarm` 做超时，避免卡死

因此当你在数据里用 `<|python_start|>` 包裹表达式时，请把它当成“计算器调用”，而不是“任意代码执行”。真正的代码执行沙箱在 HumanEval 评测里由 `nanochat/execution.py` 负责（见 10.6）。

**Rule-of-thumb：** 工具系统最常见的失败原因不是“模型不会算”，而是“格式不对齐”：训练数据与推理引擎在空格/换行/边界 token 上出现字节级偏移。排障时优先对齐 token 序列，而不是凭感觉调温度。

---

## 10.4 CLI：`chat_cli.py` 的最小交互闭环

CLI 是最轻量的调试入口，适合在无 UI 的 GPU 机器上快速验证：

- 对话上下文是一个 `conversation_tokens` 列表，起始是 BOS
- 每轮用户输入会被包裹进 `<|user_start|>...<|user_end|>`
- 引擎端追加 `<|assistant_start|>` 并开始生成
- 生成结束后，如果没有自然生成 `<|assistant_end|>`（例如被 `max_tokens` 截断），CLI 会手动补一个 `<|assistant_end|>`，保证上下文闭合

与 Web 版不同，CLI 会每步直接 `decode([token])` 然后打印，因此：

- 如果 tokenizer 的 decode 在某些 token 上会产生替换字符 `�`（不完整 UTF-8），CLI 可能会短暂输出乱码；
- 如果模型真的把特殊 token 当文本输出，CLI 也可能把它打印出来（Web 版会过滤终止 token，并有 UTF-8 拼接保护）。

把 CLI 当成“对齐工具”而不是“最终产品”：它最适合用来观察模型是否会输出 `<|assistant_end|>`、是否能触发 `<|python_start|>` 工具段、以及采样参数变化的即时效果。

---

## 10.5 Web：`chat_web.py` 的 worker pool、SSE 流式输出与 UTF-8 防抖

### 10.5.1 多 GPU worker pool：每张卡一份模型副本

`chat_web.py` 的并发策略是数据并行：每个 GPU 上加载一份完整模型，形成一个 WorkerPool：

- `available_workers` 是一个 `asyncio.Queue`
- 请求到来时 acquire 一个 worker，生成结束后 release 回队列
- 多 GPU 只在 CUDA 下启用（脚本会断言）

这套结构对训练/数据人员也有启发：它把“并发”与“模型状态”隔离得很干净——每个 worker 的 engine/tokenizer 都是独立的，不会互相污染 KV cache。

### 10.5.2 输入协议：Web 只支持 `user`/`assistant` 两种 role

Web 请求的 `messages` 会被转换成 token 序列：

- user：`<|user_start|> content <|user_end|>`
- assistant：`<|assistant_start|> content <|assistant_end|>`
- 最后追加一个 `<|assistant_start|>` 作为待生成起点

注意：当前 Web 实现并不接收 `system` role（虽然错误提示文本里可能会写到 system），如果你需要 system 指令，请在上层把它合并进第一条 user（这与 tokenizer 的 system 合并策略一致，见第 8 章）。

### 10.5.3 SSE 流式输出：为什么要检查 `�`

Web 端采用 Server-Sent Events（SSE）做流式输出。一个非常“细但值钱”的实现细节是：它不会把每个 token decode 后立刻发给前端，而是维护一个 `accumulated_tokens` 列表，每步都 decode 全部累计 token，并只在 decode 结果**不以替换字符 `�` 结尾**时才发送增量文本。

其动机很简单：某些 tokenizer 的 token 边界可能落在 UTF-8 的多字节字符中间，如果你在中间态把字符串发出去，前端就会看到乱码闪烁。用“累计 decode + `�` 检查”可以用很小的代价换来稳定的流式体验（decode 在这里更接近查表 + 拼接，成本相对可控）。

### 10.5.4 滥用防护：长度与采样参数的 clamp

为了避免有人用极端输入拖垮服务端，Web 实现对请求做了限制：

- **消息条数**：最多 `500` 条
- **单条消息长度**：最多 `8000` 字符
- **对话总长度**：最多 `32000` 字符（按所有 message 的字符数相加）
- **采样参数范围**：
  - `temperature ∈ [0.0, 2.0]`
  - `top_k ∈ [0, 200]`（`0` 表示“不做 top-k 截断”，在全词表采样）
  - `max_tokens ∈ [1, 4096]`

实现上这里更准确的说法是“**校验并拒绝**”而不是“clamp”：参数不在范围内会直接返回 400。另一个有趣的小细节是：role 校验只允许 `user/assistant`，但错误提示文本里会写到 `system`（这和实际行为不一致），所以你在对接时要以实现为准。

这些限制同样适用于你在内部部署时的“健康默认值”：先保证服务稳定，再谈能力上限。但也要意识到：这里限制的是**字符数**而不是 token 数——在某些 tokenizer/语言分布下，字符数并不能可靠约束 token 数；如果你需要更强的资源控制，最好把“prompt token 上限”也纳入协议。

### 10.5.5 SSE 的 payload 约定：每个 chunk 是一条 `data: ...`

在 `text/event-stream` 的 SSE 协议里，nanochat Web 端每次输出一段文本增量，就会产生一条形如 `data: ...` 的事件行，并以空行结束。payload 是 JSON：

- 增量输出：`{"token": "<新增文本>", "gpu": <gpu_id>}`
- 流结束：`{"done": true}`

一个简化的示例如下：

```text
data: {"token":"你好","gpu":0}

data: {"token":"，我能帮你什么？","gpu":0}

data: {"done": true}
```

注意实现里用了 `ensure_ascii=False`，因此中文不会被转义成 `\\u4f60\\u597d` 这种形式；再配合上一节的 UTF-8 “`�` 防抖”，前端体验会稳定很多。

### 10.5.6 “流式优先”的工程含义：API 只有 streaming

当前 `/chat/completions` 端点只提供 streaming 形态：服务端边生成边推送，直到发出 `{"done": true}`。这会影响你做离线评测/批处理的方式：如果你想要“非流式一次性返回”，要么在客户端把 stream 消费并拼接，要么你需要在服务端加一个“收集完整文本再返回”的分支（同时仍要记得在 `finally` 中 release worker，避免 worker 泄漏导致服务逐渐拒绝请求）。

---

## 10.6 评测中的代码执行：HumanEval 的沙箱不是安全沙箱

nanochat 的 HumanEval 评测会把模型生成的代码提取出来，并用 `nanochat/execution.py` 在子进程里执行（带超时与资源限制）。这套机制能防止大多数“意外破坏”（例如无限循环、误删文件等），但它**不是**真正的安全沙箱：

- **执行是独立进程**：每次运行在单独的 `multiprocessing.Process` 中，主进程可在超时时 kill。
- **默认限制**：timeout 默认 `5s`；内存上限默认 `256MB`（注意：在 macOS/Darwin 上，资源限制分支会被跳过，表现会不同）。
- **“评测友好”的解析**：`tasks/humaneval.py` 会优先提取第一个 Markdown 代码块（```python 或 ```），否则把整个 completion 当代码。这意味着“带少量包装”的输出通常还能评测，但最稳妥的还是训练模型输出**纯代码**、不夹解释文本。
- 网络访问不一定被阻断
- Python 动态特性可能绕过部分限制
- 没有内核级隔离（seccomp/容器/虚拟化）

**Rule-of-thumb：** 在你不完全信任模型输出时，不要在宿主机上裸跑代码执行评测；把它放进容器/隔离环境里当成“高风险操作”对待。

---

## 本章小结

- Engine 是 token 级推理：prefill 一次、decode 多步；KV cache 布局为 FA3 友好形状，并用 `cache_seqlens` 追踪位置。
- nanochat 的采样只有 `temperature` 与 `top_k`；`temperature=0` 时是确定性贪心解码，`top_k` 不起作用。
- 工具调用由 `<|python_start|>...<|python_end|>` 触发，引擎计算并强制注入 `<|output_start|>...<|output_end|>`；forced token 会被标记 mask=0，便于训练侧剔除。
- CLI 是最小闭环调试工具；Web 版使用 worker pool + SSE 流式输出，并用 `�` 检查避免 UTF-8 乱码闪烁。
- HumanEval 的执行环境是“评测用保护”，不是安全沙箱；需要额外隔离。

---

## 练习题

1. **（基础）为什么 Engine 要先做 batch=1 的 prefill，再把 KV cache 复制成 `num_samples` 份，而不是直接 batch=`num_samples` 做 prefill？**
   - *Hint：prompt 的计算能不能复用？prefill 成本在哪？*
   <details>
   <summary>查看提示与答案</summary>
   prefill 的主要成本是对长 prompt 的一次性前向计算。batch=1 prefill 只算一次 prompt，然后复制 cache 就能让多条样本共享同一份 prompt 计算结果；如果直接 batch=num_samples prefill，就会重复计算同一 prompt num_samples 次，浪费大量算力与显存带宽。
   </details>

2. **（基础）解释 `temperature=0` 时 `top_k` 为什么不影响结果。**
   - *Hint：实现里是否进入采样分支？*
   <details>
   <summary>查看提示与答案</summary>
   因为温度为 0 时引擎走的是贪心解码（argmax）分支，不会做 multinomial 采样，也不会执行 top-k 截断逻辑。此时输出完全由最大 logit 决定，`top_k` 参数不会改变路径。
   </details>

3. **（挑战）KV cache 显存为什么会在 decode 阶段随 “并发样本数 B” 与 “上下文长度 L” 线性增长？请写出一个近似的量级公式。**
   - *Hint：cache 需要为每层存 K 和 V。*
   <details>
   <summary>查看提示与答案</summary>
   KV cache 需要为每层、每个样本、每个位置、每个 KV head 存储 key 与 value 两个张量，因此显存大致与 `2 * n_layers * B * L * n_kv_head * head_dim * bytes_per_element` 成正比（最前面的 2 来自 K 与 V）。因此并发样本数和最大生成长度都是推理 OOM 的一线旋钮。
   </details>

4. **（挑战）工具调用状态机：当模型输出 `<|python_start|>`、一段表达式 token、再输出 `<|python_end|>` 时，引擎会注入哪些 token？这些 token 的 mask 应该是多少？**
   - *Hint：forced token 队列。*
   <details>
   <summary>查看提示与答案</summary>
   引擎会把表达式解码并执行计算器；若得到结果，会把 `<|output_start|>`、结果字符串对应的 token、`<|output_end|>` 依次压入 forced 队列。随后这些 token 会被强制输出，因此它们的 mask 应为 0（表示不是模型采样得到，而是系统注入/强制）。
   </details>

5. **（挑战）为什么 Web 端要“累计 token 再 decode”，并检查解码结果是否以 `�` 结尾？**
   - *Hint：多字节 UTF-8 字符可能被拆开。*
   <details>
   <summary>查看提示与答案</summary>
   某些 tokenizer 的 token 边界可能落在 UTF-8 多字节字符中间。若逐 token decode 并立即发送，前端会短暂看到不完整字符（通常显示为替换字符 `�`）造成闪烁。累计 token 并只在字符串末尾不是 `�` 时发送增量文本，可以避免把中间态发出去，从而稳定流式体验。
   </details>

6. **（挑战）WorkerPool 的 acquire/release 机制解决了什么问题？当所有 worker 都忙时会发生什么？**
   - *Hint：并发与状态隔离。*
   <details>
   <summary>查看提示与答案</summary>
   它把“并发请求调度”与“每张 GPU 上的模型/engine 状态”隔离开：每个 worker 独占一份模型副本与推理状态，避免 KV cache 等状态被跨请求污染。当所有 worker 都忙时，请求会在 `available_workers` 队列上等待，直到有 worker 被 release 回来。
   </details>

7. **（挑战）为什么 HumanEval 的执行环境被强调“不是安全沙箱”？对内部评测与对外服务分别有什么建议？**
   - *Hint：对抗性代码与宿主机风险。*
   <details>
   <summary>查看提示与答案</summary>
   因为它主要防的是“意外破坏”，并没有提供内核级隔离；对抗性代码可能绕过限制、访问网络或做更复杂的逃逸。内部评测也应放进容器/隔离环境；对外服务更不应执行不可信代码，至少需要更严格的沙箱（容器、seccomp、网络隔离、只读文件系统等）。
   </details>

---

## 常见陷阱与错误 (Gotchas)

1. **把 top-p 当成可用旋钮**
   - **症状**：你按其他框架习惯去设置 top-p，发现 nanochat 没效果或压根没有参数入口。
   - **修复**：nanochat 采样只实现 `temperature` 与 `top_k`；需要 top-p 就得改引擎采样逻辑，并同步调整训练/评测的默认假设。

2. **Web 请求里塞 `system` role**
   - **症状**：请求被拒绝或校验失败。
   - **修复**：当前 Web 只支持 `user/assistant`。需要 system 指令时，把它合并进第一条 user（与 tokenizer 的 system 合并策略对齐）。

3. **KV cache OOM：并发与最大生成长度没控住**
   - **症状**：decode 阶段突然 OOM，尤其在多样本采样或长对话时。
   - **修复**：优先降低 `max_tokens`/并发样本数（Web 的并发由 worker 数决定，RL/评测的并发由 batch 决定），必要时降低上下文长度或模型规模。

4. **工具调用“格式不对齐”导致状态机失灵**
   - **症状**：模型输出了 `<|python_start|>` 但引擎没注入结果，或注入后模型继续重复调用。
   - **修复**：先确认模型确实输出了 `<|python_end|>`（引擎在 end 时才 eval）；再检查表达式 token 解码后是否落在计算器允许的字符集合内；最后对齐训练样本与推理拼接在空格/换行上的分布。

5. **把 HumanEval 执行当成“安全运行环境”**
   - **症状**：在宿主机上直接执行模型生成代码，担心系统被破坏。
   - **修复**：把它当成评测辅助而不是安全沙箱。任何不可信代码执行都应放在更严格的隔离环境里。

6. **推理端 dtype 变更但忘了 KV cache**
   - **症状**：你改了推理端 autocast/dtype（或在 CUDA 上强制 FP32），结果要么直接报 dtype 不匹配，要么吞吐/数值表现怪异。
   - **修复**：检查 `Engine.generate()` 里 KV cache 的 dtype 假设（CUDA=BF16，其他=FP32）。如果你要改变推理精度策略，需要把 KV cache 的分配逻辑一起改掉，或改成延迟分配/显式传参。
