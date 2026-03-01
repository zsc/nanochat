# 第 8 章：SFT 数据准备（对话格式、render/mask、任务混合、best-fit pad、identity 与工具）

预训练把模型变成了“强大的文本压缩器”，但它并不会天然以“助手”的方式对话：它不知道什么时候该等用户、什么时候该停止、也不知道哪些内容应该由工具计算、哪些内容应该由语言生成。监督微调（SFT）的任务，就是把这种纯语言建模能力，塑形成一套**可控的交互协议**：固定的对话边界、固定的结束信号、固定的工具调用形态、以及可复现的任务混合策略。

nanochat 的 SFT 数据准备有一个很鲜明的工程特征：它不追求“通用对话模板兼容一切生态”，而是把协议压到最小、写进 tokenizer 和训练脚本里。对数据准备人员来说，这既是好消息（约束少、可控性强），也是危险点（任何细节不一致都会带来分布偏移）。本章会沿着代码的实际路径，把它拆成三层：**对话对象长什么样**、**如何渲染成 token 序列并生成 mask**、**训练时如何 packing 与混合任务**，并把 nanochat 里一些“很有意思的细节”嵌到合适的位置（例如 GSM8K 的工具轨迹、TaskMixture 的确定性 shuffle、以及多选题渲染时一个空格就能毁掉小模型的绑定能力）。

---

## 8.1 Conversation 数据模型：`messages`、`parts` 与“严格交替”

在 nanochat 的 `tasks/` 里，“一条训练样本”统一被表示为一个 conversation 字典：

- `conversation["messages"]` 是一个消息列表
- 每条消息是 `{ "role": "...", "content": ... }`
- role 主要是 `user` / `assistant`（tokenizer 也支持 conversation 第一条是 `system`，但有额外处理，见 8.2）

最关键的强约束是：**消息必须严格交替**。

`nanochat/tokenizer.py` 的 `render_conversation()` 会做断言：

- 第 0 条（去掉 system 处理后）必须是 `user`
- 第 1 条必须是 `assistant`
- 之后依次交替

这条约束看似死板，但它把一类非常常见的“静默错误”提前变成了显式崩溃：如果你数据里出现了 `user,user,assistant` 这种结构，很多框架会默默拼接成怪异 prompt，导致模型学到“自问自答”的坏习惯；nanochat 选择直接 crash，让你在数据准备阶段就修掉。

另外，assistant 的 `content` 有两种形态：

1. **纯字符串**：最常见的对话回复。
2. **parts 列表**：用于表达“文本 + 工具调用 + 工具输出”的轨迹。每个 part 是形如 `{ "type": "...", "text": "..." }` 的字典，常见 type 包括：
   - `text`
   - `python`（工具调用：表达式）
   - `python_output`（工具输出：结果字符串）

这种 parts 结构不是为了“好看”，而是为了让渲染阶段能精确插入工具边界 token，并为后续的 supervision mask 做准备。

---

## 8.2 nanochat 的对话渲染：`render_conversation()` 与特殊 Token 协议

SFT 的第一性原理是：**训练时喂给模型的 token 序列，必须与推理时引擎构造的 token 序列一致**。nanochat 把这条一致性写成了一个非常明确的特殊 token 协议（见 `SPECIAL_TOKENS`）：

- `<|bos|>`：每个对话/文档的起始锚点（BOS）
- `<|user_start|>` / `<|user_end|>`：包裹用户消息
- `<|assistant_start|>` / `<|assistant_end|>`：包裹助手消息
- `<|python_start|>` / `<|python_end|>`：包裹工具调用表达式（由 assistant 生成）
- `<|output_start|>` / `<|output_end|>`：包裹工具输出（在推理时由引擎注入）

### 8.2.1 System 消息的特殊处理：合并到第一条 user

`render_conversation()` 支持 conversation 的第 0 条消息为 `system`，但它并不会单独渲染一段 `<|system_start|>`（nanochat 没有这个 token）。它做的是一种“外科手术”：

- 断言 system 后必须紧跟一条 user
- 把 system 的内容与 user 的内容用 `"\n\n"` 拼起来
- 然后把 system 消息丢掉，后续按 user/assistant 交替渲染

这对数据准备很重要：如果你想引入 system 指令，请意识到它会被“折叠”进 user 文本里，并且中间必然出现两个换行符。

### 8.2.2 扁平 token 流的物理形态（ASCII 展开）

下面用一个带工具的简化对话，展示它在 token 序列里的“物理展开”。注意：这是概念图，真实 tokenization 还会把文本切成 BPE token id。

```text
[BOS]
 <|user_start|>   用户问题文本...   <|user_end|>
 <|assistant_start|>
   助手自然语言... 
   <|python_start|>  12/60  <|python_end|>
   <|output_start|>  0.2    <|output_end|>
   ...继续推理，最终写出 #### 10
 <|assistant_end|>
 (下一轮 user/assistant ... )
```

其中 `<|assistant_end|>` 非常关键：它是一个“硬停止信号”，推理引擎会把它作为 stopping criteria 之一（见第 10 章）。如果你的 SFT 数据里缺少它，模型就很难学会“何时闭嘴”。

### 8.2.3 `mask` 的语义：哪些 token 应该产生监督信号

`render_conversation()` 会返回两个等长数组：

- `ids`：token id 序列
- `mask`：0/1 序列，`mask=1` 表示该位置的 token 期望被监督（产生 loss/梯度）

nanochat 的 mask 设计体现了一个非常清晰的意图：

- user 侧 token（包括 `<|user_start|>`、用户文本、`<|user_end|>`）全部 `mask=0`
- assistant 侧的自然语言与工具调用表达式（`<|python_start|>...<|python_end|>`）是 `mask=1`
- 工具输出（`<|output_start|>...<|output_end|>`）是 `mask=0`，因为这些 token 在推理时来自“外部系统”（引擎注入），不应该让模型去学会“凭空猜工具输出”
- `<|assistant_end|>` 被标为 `mask=1`（模型需要学会输出它来停止）

这套 mask 让你可以在数据层面明确表达：“模型负责说话与发起工具调用；工具输出由系统注入，不监督。”

### 8.2.4 `render_for_completion()`：评测与 RL 的关键拼接点

在评测与 RL 中，conversation 通常包含“标准答案的 assistant 消息”（例如 GSM8K 的参考解答）。但生成时我们需要让模型补全一个新的 assistant 回复，而不是把参考答案继续喂给它。

因此 nanochat 提供了 `render_for_completion()`：

- 深拷贝 conversation
- **删除最后一条 assistant 消息**
- 用 `render_conversation()` 渲染剩余内容
- 在末尾追加 `<|assistant_start|>`，把模型置于“等待生成 assistant”状态

`scripts/chat_eval.py`、`scripts/chat_rl.py` 都依赖这条逻辑；你若自行实现 prompt 构造，请优先复用这个语义。

---

## 8.3 SFT DataLoader：best-fit pad packing（不丢监督 token）

SFT 的对话长度同样是长尾分布。nanochat 在 `scripts/chat_sft.py` 里实现了一个与 base 预训练“同一风格”的 best-fit，但关键取舍相反：**宁愿 padding，也不裁剪**。

它的核心策略是：

1. 维护一个 conversation token 序列缓冲（每条对话已经包含 BOS 与各种边界 token）
2. 构造每一行容量为 `max_seq_len+1`
3. 优先选择“能完整放入的最长对话”拼进去（best-fit）
4. 若剩余空间装不下任何完整对话，就用 BOS token 把剩余填满（pad），并把 padding 对应的 targets 设为 `-1`

这里的 `-1` 是 nanochat 的 ignore_index 约定（`nanochat/gpt.py` 的交叉熵就是 `ignore_index=-1`）。这与很多默认使用 `-100` 的代码库不同，是一个需要全链路对齐的约定。

### 8.3.1 一个容易忽略但很关键的事实：当前 SFT 训练只 mask 了 padding

尽管 tokenizer 能生成 `mask`（用于标记 user token 与工具输出 token），但 `chat_sft.py` 当前的数据生成器只使用 `ids`，并没有把 `mask` 转换成 targets 的 ignore_index。它主要只做了两件事：

- packing + BOS padding
- 把 padding 位置的 targets 设为 `-1`

这意味着：**SFT 实际上更接近“对整段对话 transcript 做 next-token modeling”**，而不是经典的“只训练 assistant 回复”。这不是对错问题，而是你必须知道的事实：它会影响你对 loss 的解读、也会影响工具输出 token 是否被监督。

如果你希望严格遵循 `render_conversation()` 的 mask 语义（不监督 user 与工具输出），你需要在数据准备或 loss 计算处显式应用这个 mask（把 mask=0 的位置置为 `-1`）。这类改造属于“配方变更”，会改变收敛与行为，请在小模型上先做对照实验再放大。

### 8.3.2 DDP 视角：SFT 的分片与“结束条件”要跨 rank 同步

SFT 训练的 dataset 来自 `TaskMixture`（见 8.4），每个 rank 按 `cursor += world_size` 的 stride 读取不同样本，以避免重叠。由于生成器内部还有 buffering（提前 refill），`chat_sft.py` 会单独追踪“实际 consumed 的样本数”，并用它估计进度与 epoch。

更关键的是：当某个 rank 认为 “last_step=True”（例如跑完一轮数据或达到 `--num-iterations`），其他 rank 必须同步停下，否则就会出现分布式 hang。脚本会用一次 all-reduce 把 `last_step` 在所有 rank 上取 max，保证大家一起结束。

---

## 8.4 任务混合：TaskMixture 的确定性 shuffle 与 oversample

SFT 的能力来自“混合”：闲聊对话教语气与礼貌，多选题教约束输出空间，GSM8K 教推理与工具，拼写类教极端细粒度的字符感知……但混合的关键不是“把数据拼一起”，而是：

1. **顺序要可复现**：否则你无法解释某次 run 的波动来自哪里；
2. **过采样要可控**：小而关键的数据必须被看见足够多次。

`tasks/common.py` 的 `TaskMixture` 做了一个非常工程化的选择：

- 它把所有任务的所有样本展开成一个 `(task_idx, local_idx)` 的 index_map
- 然后用固定种子（`Random(42)`）做一次全局 shuffle

因此，只要任务本身的数据是确定的，混合顺序就是确定的。更妙的是：**过采样不需要写复杂权重**——你只要把同一个 task 对象在列表里放多次，它的样本就会在 index_map 里出现多次，相当于多 epoch。

`scripts/chat_sft.py` 里的训练混合（大意）包括：

- SmolTalk（大量通用对话）
- identity conversations（`identity_conversations.jsonl`，放两次以实现 2 个 epoch）
- MMLU（多选题，按 `mmlu_epochs` 次重复）
- GSM8K（数学推理 + 工具轨迹，按 `gsm8k_epochs` 次重复）
- 拼写类任务（极端细粒度监督）

**Rule-of-thumb：** 混合时不要只看“样本条数比例”，更要看“token 数比例”。一个很长的对话样本，会在 token 维度上自然占更大权重。

---

## 8.5 教模型用工具：GSM8K 的 `<< >>` 与 python/tool token

GSM8K 的参考答案文本里包含形如 `<<12/60=0.2>>` 的工具调用痕迹。nanochat 的 `tasks/gsm8k.py` 会把它解析成 parts 序列：

- `python` part：表达式（例如 `12/60`）
- `python_output` part：结果（例如 `0.2`）

渲染阶段，这会变成 `<|python_start|>...<|python_end|>` 与 `<|output_start|>...<|output_end|>` 的结构（见 8.2）。推理阶段（第 10 章）引擎会在模型生成 `python_start ... python_end` 后执行一个“安全计算器”，并把结果以 `<|output_start|>...<|output_end|>` 的形式强制注入回上下文。

这条链路的关键不是“工具很强”，而是**格式必须一致**：

- 训练数据里工具调用的边界 token、换行、空格、是否有逗号，都属于模型会学习的分布；
- 推理引擎注入的 token 必须与训练分布对齐，否则模型会把注入结果当成“乱码”，继续尝试自己生成或重复调用工具。

**有意思的小细节：** 多选题渲染里甚至会因为一个空格导致小模型崩掉。`tasks/common.py` 的 `render_mc()` 特意强调：在 `choice=letter` 里 **`=` 和字母之间不留空格**，因为 tokenizer 会把 `" A"` 与 `"A"` 切成不同 token；而 assistant 的答案通常是单个字母 `"A"`（无前导空格）。这种字节级对齐对小模型尤其重要。

---

## 8.6 Identity 对话：让模型“知道自己是谁”

nanochat 的 SFT 混合里包含一份 identity 对话文件：`identity_conversations.jsonl`，默认路径是 `NANOCHAT_BASE_DIR/identity_conversations.jsonl`。它由 `tasks/customjson.py` 的 `CustomJSON` 读取：每一行是一段 JSON 数组，数组里是交替的 `user/assistant` 消息。

identity 数据的价值是“低成本但高杠杆”：

- 它能显著降低模型在身份问题上的幻觉（例如被诱导说“我是某某商业模型”）
- 它能把模型的自我介绍、能力边界、拒答语气固定下来

但它也很容易被用坏：

**Rule-of-thumb：**

1. **少量、多次、均匀混入**：identity 不需要很多；更重要的是不要集中在训练开头/结尾。nanochat 通过“把该 task 放两次”来做轻量过采样，本质上就是让它在全局 shuffle 里均匀出现更多次。
2. **对抗性覆盖**：不要只写“你好我是 nanochat”；要覆盖诱导、挑衅、越权等边界场景，让模型学会稳定拒绝与澄清。
3. **保持短而稳定**：identity 往往是高频触发场景，短模板更稳定、更容易被小模型学牢。

---

## 本章小结

- nanochat 的 SFT 样本是 conversation：`messages` 严格交替；assistant 可包含 parts（文本/工具调用/工具输出）。
- `render_conversation()` 把 conversation 渲染为特殊 token 协议：`<|user_start|>...<|user_end|>`、`<|assistant_start|>...<|assistant_end|>`，工具边界用 `<|python_*>` 与 `<|output_*>`。
- tokenizer 能返回监督 `mask`（不监督 user 与工具输出）；`render_for_completion()` 用于评测/RL 的“删参考答案 + 追加 `<|assistant_start|>`”。
- `chat_sft.py` 的 packing 是 best-fit **pad**：不裁剪，padding 用 BOS，并把 padding targets 设为 `-1`（nanochat 的 ignore_index 约定）。
- TaskMixture 用固定种子做全局 shuffle；过采样通过“重复加入 task”实现，简单且可复现。
- 工具与多选这类能力对“字节级格式一致性”极其敏感，一个空格都可能导致小模型绑定失败。

---

## 练习题

1. **（基础）为什么 nanochat 强制对话消息严格交替（user/assistant/user/assistant…）？如果你把两条 user 消息连续放在一起，会发生什么？**
   - *Hint：想想 `render_conversation()` 的断言为什么比“默默拼接”更安全。*
   <details>
   <summary>查看提示与答案</summary>
   严格交替能把“坏格式数据”尽早暴露出来，避免静默分布污染。若出现连续 user 消息，nanochat 会在渲染阶段断言失败并 crash，让你回到数据准备阶段修复；如果默默拼接，模型可能学到“自问自答/乱切角色边界”的坏习惯，后果很难从 loss 上直接看出。
   </details>

2. **（基础）写出 nanochat 一轮对话在 token 流里的边界结构（无需展开 BPE 细节）：包含 BOS、user/assistant 边界、以及 assistant 结束信号。**
   - *Hint：用 `<|bos|>`、`<|user_start|>`、`<|user_end|>`、`<|assistant_start|>`、`<|assistant_end|>`。*
   <details>
   <summary>查看提示与答案</summary>
   一轮对话的边界结构是：
   `[BOS] <|user_start|> 用户文本 <|user_end|> <|assistant_start|> 助手文本 <|assistant_end|>`。
   若有多轮，就重复 user/assistant 片段，并保持每条 user/assistant 都被对应的 start/end token 包裹。
   </details>

3. **（基础）`render_for_completion()` 为什么要“删除最后一条 assistant 消息”，并在末尾追加 `<|assistant_start|>`？**
   - *Hint：conversation 里最后一条 assistant 往往是参考答案。*
   <details>
   <summary>查看提示与答案</summary>
   因为在评测/RL 中，conversation 通常包含参考答案的 assistant 消息；若直接渲染整段对话，模型只是在继续参考答案之后的文本，不是在“回答问题”。删除最后一条 assistant 后再追加 `<|assistant_start|>`，相当于把模型置于“该轮 assistant 需要生成”的起点，才能公平评测与采样。
   </details>

4. **（基础）为什么 `chat_sft.py` 选择“best-fit + padding”而不是像 base 预训练那样“best-fit + cropping”？**
   - *Hint：监督 token 的稀缺性与价值。*
   <details>
   <summary>查看提示与答案</summary>
   base 预训练数据量巨大，丢掉一部分 token 可以接受，换取 100% 利用率；而 SFT 数据通常更稀缺、更昂贵（人类/高质量合成/任务数据），裁剪会直接丢监督信号，得不偿失。因此 SFT 更愿意 padding 并用 ignore_index 把 padding 的 loss 排除。
   </details>

5. **（挑战）best-fit pad packing：给定 `row_capacity=11`，缓冲里有对话长度 A(6)、B(5)、C(4)。按 best-fit pad 的策略，第一行会如何填充？哪些 target 位置会被置为 `-1`？**
   - *Hint：先放能完整放入的最长对话；放不下任何完整对话时用 BOS padding，并从“预测第一个 padding token”开始 mask。*
   <details>
   <summary>查看提示与答案</summary>
   先放 A(6)，剩余 5；再放 B(5)，剩余 0，因此第一行是 A+B，完全无 padding。targets 不需要置 `-1`（除非实现里有额外 mask 规则）。如果把缓冲换成 A(6)+C(4)，剩余 1 放不下任何完整对话，就会 pad 1 个 BOS；此时从 `content_len-1` 起的 targets 会被置为 `-1`，确保不训练去预测 padding。
   </details>

6. **（挑战）解释为什么 `render_mc()` 要刻意保证 `=` 与选项字母之间没有空格。这类“空格敏感”问题在小模型上更严重的原因是什么？**
   - *Hint：`" A"` 与 `"A"` 的 tokenization 可能不同；小模型的泛化更弱。*
   <details>
   <summary>查看提示与答案</summary>
   tokenizer 可能把 `" A"`（带前导空格）与 `"A"` 切成不同 token id，而多选题评测往往要求 assistant 输出单个字母 token（无前导空格）。如果 prompt 里出现的是带空格的版本，模型学到的绑定会偏移，导致在评测时输出不稳定。大模型有更强的鲁棒性，小模型更依赖“字节级一致性”，因此问题更严重。
   </details>

7. **（挑战）工具调用轨迹对齐：如果训练数据里工具输出被监督了（模型学会自己生成 `<|output_start|>...`），而推理引擎又会强制注入工具输出，会出现哪些行为问题？你会如何用 mask 设计去避免？**
   - *Hint：想想“重复输出/分布偏移/工具输出幻觉”。*
   <details>
   <summary>查看提示与答案</summary>
   可能的问题包括：模型在没有调用工具时也幻觉性输出 `<|output_start|>` 段；调用工具后输出被注入但模型仍试图再生成一次导致重复；或模型把注入结果当成普通文本继续续写。要避免这类分布混乱，最稳妥的做法是遵循 `render_conversation()` 的语义：将工具输出段设为不监督（mask=0），并在训练时把这些位置的 targets 置为 `-1`，让模型学会“输出由系统注入，不由我预测”。
   </details>

---

## 常见陷阱与错误 (Gotchas)

1. **system 消息被悄悄合并到第一条 user**
   - **症状**：你以为自己在训练“system 指令”，但实际它被折叠进 user 文本；或者你在数据里同时写了 system 和 user 的固定前缀，导致重复。
   - **修复**：理解 `render_conversation()` 的规则：system 只允许出现在第 0 条，并会与第 1 条 user 用 `\\n\\n` 合并。设计模板时避免双重注入。

2. **以为 SFT 只训练 assistant，实际脚本只 mask 了 padding**
   - **症状**：你用 “只算 assistant token 的有效长度” 去解释 loss，发现对不上；或者工具输出 token 被模型学会“自己生成”。
   - **修复**：明确当前 `chat_sft.py` 的实现事实；若你要严格按 mask 训练，需要自己改造数据/损失，并先做小规模对照实验。

3. **工具输出格式不一致导致推理时状态机失灵**
   - **症状**：模型能生成 `<|python_start|>`，但后续行为混乱：重复调用、忽略注入结果、或生成乱码边界 token。
   - **修复**：训练与推理必须一致：同样的边界 token、同样的空格/换行分布。调试时优先把“引擎实际喂给模型的 token 序列”与训练样本做逐字符 diff。

4. **长对话被截断，关键结束符消失**
   - **症状**：训练后模型不愿意输出 `<|assistant_end|>`，或评测中频繁因为格式缺失而判错。
   - **修复**：注意 `render_conversation(max_tokens)` 是简单的切片截断，可能切掉末尾的 `<|assistant_end|>` 或关键答案段。对超长样本做过滤/重写，或提高 `max_seq_len` 并重新做配方对齐。

5. **identity 文件缺失导致混合数据静默变少**
   - **症状**：SFT 启动时控制台出现 warning，或 identity 数据实际上没被加载；模型身份问题频繁幻觉。
   - **修复**：确保 `NANOCHAT_BASE_DIR/identity_conversations.jsonl` 存在且每行格式正确（交替 user/assistant 字符串消息）。把 identity 当成“高杠杆小数据”，但不要指望它在缺失时还能自动补救。
