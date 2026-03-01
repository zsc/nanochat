# nanochat_book：lessons learnt（从“生成文档”到“与代码严格对齐”）

这份文档记录从本书启动以来的一些“可复用经验”，重点聚焦：**如何把中文教程写到与代码实现一致**，并把仓库里那些“细但值钱”的工程细节放进合适章节，避免常见的 LLM 文档幻觉与跨仓库惯性。

---

## 1) 把“代码”当成唯一事实源（single source of truth）

最重要的一条：**不要用你对别的 LLM 训练栈的记忆来推断 nanochat**。nanochat 的实现有不少“与主流默认不同”的硬约定；一旦写错，读者按文档做就会直接踩坑，而且往往是静默错误。

我们最后采取的写作策略是：

- **每一条关键结论都能落到具体文件**（至少落到 `nanochat/` 或 `scripts/` 的某段实现）。
- 对“可能变化/有歧义”的点，优先写成**实现事实**（例如常量范围、停止条件、mask 语义），而不是泛泛而谈。
- 把“代码中的约定/协议”当成文档主线（对训练与数据准备人员最有价值的恰恰是这些约定）。

典型例子（都是一开始很容易写错的）：

- `ignore_index` 在 nanochat 是 `-1`（不是很多仓库默认的 `-100`）：见 `nanochat/gpt.py`、`scripts/chat_sft.py`、`scripts/chat_rl.py`、`nanochat/loss_eval.py`。
- 采样只实现 `temperature` + `top_k`，没有 top-p：见 `nanochat/engine.py` 与 `nanochat/gpt.py`。
- Base packing 的策略是 BOS-aligned best-fit **cropping**，不做 block-diagonal attention mask：见 `nanochat/dataloader.py`。
- 对话/工具的控制 token 协议是 `<|user_start|>...` 这套，不是 ChatML：见 `nanochat/tokenizer.py`、`nanochat/engine.py`、`scripts/chat_cli.py`、`scripts/chat_web.py`。

---

## 2) 与代码结合的“抓手”：先找协议与边界，再写原理与经验

对训练/数据人员来说，最需要被讲清楚的往往不是“Transformer 的通用知识”，而是 **这个仓库的协议边界**：

- 什么是输入？（token ids 还是文本？谁负责渲染模板？）
- 什么是监督目标？（targets 从哪来？哪些位置被 ignore？）
- 什么是停止条件？（推理何时停？终止 token 是什么？）
- 什么算指标？（BPB 统计里哪些 token 不算？忽略 targets 如何处理？）

因此我们在写每章时，都先定位“边界协议”，再讨论动机与 trade-off。实践上，最有效的阅读路径是：

1. 从 `runs/speedrun.sh` 找到端到端“事实标准”的入口。
2. 沿着 `scripts/*.py` 看每阶段的 I/O 约定（训练、SFT、RL、eval、web）。
3. 再回到 `nanochat/*.py` 抽象层理解实现细节（tokenizer / dataloader / engine / checkpoint / attention / fp8）。

这个顺序的好处是：你写出来的文档更像“可落地的工程手册”，而不是“概念综述”。

---

## 3) LLM 初稿可以提速，但必须用“系统化对齐”去消除幻觉

我们确实用 `generate_with_gemini.py` + `template.md` 快速生成了章节初稿，但早期很快遇到一个规律：

- LLM 很容易把别的栈的默认设定“顺手写进来”（例如 ChatML、top-p、EOS 终止符、`apply_chat_template`、block-diagonal attention mask、`-100` ignore_index、PPO/KL 之类 RL 术语）。
- 这些内容写得再顺也没用：只要不在代码里，就必须删掉或改写成 nanochat 的真实实现。

最后形成的稳定工作流是：

- **生成**：让 LLM 写“结构完整、叙事流畅”的初稿（章节结构、练习题、Gotchas 都齐）。
- **对齐**：逐小节用 `rg` + 打开源码文件核对，把“泛化叙述”改成“实现事实 + 工程动机”。
- **扫雷**：全局 grep 一遍禁止词（例如 `ChatML`、`im_start`、`top-p`、`apply_chat_template`、`-100` 等），防止漏网之鱼在后面章节出现。

关键心法：**让 LLM 当写作助手，不当事实来源**。

---

## 4) 最有价值的“有意思细节”，往往来自小而硬的实现

很多真正能帮训练/数据同学排障的内容，不是大理论，而是那些你一旦知道就能省一周的“小细节”。这类细节的共同特征是：它们通常只在代码里出现，不会在 README 里被反复强调。

我们把这些细节尽量编织进对应章节，典型包括：

- `nanochat/engine.py` 的工具状态机：`<|python_start|>...<|python_end|>` 触发，输出段 `<|output_start|>...<|output_end|>` 是 **forced token**，并通过 `token_masks` 标为 0（这直接决定 RL/SFT 的监督边界）。
- `scripts/chat_web.py` 的 SSE 流式输出：累计 decode 并检查末尾是否为替换字符 `�`，避免多字节 UTF-8 在 token 边界被拆开导致前端闪烁乱码。
- `nanochat/flash_attention.py` 的 FA3/SDPA 自动切换：FA3 只在 Hopper(sm90) 上；Ada/Blackwell/MPS/CPU 都走 SDPA fallback。排障吞吐必须先确认“实际跑的是哪个实现”。
- `nanochat/checkpoint_manager.py` 的 `_orig_mod.` 前缀修复：torch.compile 与非 compile 之间切换时能救命。
- `nanochat/common.py` 的 `download_file_with_lock()`：多 rank 下载评测包用 file lock 防互踩；前提是所有进程的 `NANOCHAT_BASE_DIR` 真一致。
- Base dataloader 的 DDP row group stride + 近似 resume：保存的是 `pq_idx/rg_idx/epoch`，恢复会“前进一个分片”避免重放，强调“近似而非精确”。

写作上，我们总结出一个判断准则：

> 如果一个细节会导致“训练看似正常但能力很差”或“线上服务偶发炸/乱码/卡死”，它就应该进书，而且最好放进 Gotchas。

---

## 5) 与代码强绑定的“排障写法”：把每个 gotcha 写成可执行的检查

“Gotchas”不是吐槽区，最好写成可操作的检查项。我们在后期调整为固定写法：

- **症状**：读者能观察到什么（loss 异常、输出乱码、服务 400、OOM、吞吐锯齿）。
- **根因（实现事实）**：哪段代码/哪条协议导致这种现象。
- **修复**：最小修改/最小验证路径是什么。

例如：

- “把 top-p 当成可用旋钮” → 根因是 `Engine` 只实现 `temperature/top_k` → 修复是改采样或改预期。
- “Web 请求里塞 system role” → 根因是 `scripts/chat_web.py` 只接受 `user/assistant` → 修复是 system 合并到第一条 user。
- “工具输出被算进 loss” → 根因是 forced token + SFT 默认不使用 render mask → 修复是显式剔除或修改训练脚本使用 mask。

这种写法的好处是：读者能直接拿着 gotcha 去定位，而不是把它当成“经验之谈”。

---

## 6) 一套通用的“文档一致性检查清单”（更新代码时也能复用）

当仓库代码更新时，文档最容易悄悄漂移。我们最后沉淀出一套很省心的检查顺序：

1. 更新 `index.md` 与 `template.md` 里的对齐 commit（例如 `git rev-parse --short HEAD`）。
2. 章节体量检查：每章目标约 20KB（`wc -c chapter*.md`）。
3. 全局扫雷：`rg` 查禁用术语与错误协议（ChatML/top-p/-100/EOS/im_start…）。
4. 关键协议再核对一次：
   - special tokens 列表
   - ignore_index 值
   - sampling 参数
   - stop tokens
   - base/sft/rl 的 packing 与 mask 语义
   - web 限额与 role
5. 如果仓库包含被 git 管理的衍生产物（例如 `html/`），不要误删；必要时用 `git checkout -- <dir>` 恢复，再决定是否需要重新生成。

本质上，这是把“文档维护”做成一个轻量的 release checklist。

---

## 7) 最终结论：写“能跑通”的文档，比写“懂理论”的文档更难也更值钱

这次工作最大的体会是：面向训练与数据人员的文档，价值不在于“把通用理论讲得多优雅”，而在于把仓库里真正决定成败的几条硬约定讲清楚，并且让每个关键结论都能回到代码验证。

当你做到这一点，文档会自然具备三个特征：

- **可复现**：读者按书走能跑出同样的产物与指标。
- **可排障**：遇到异常能沿着“协议 → 实现 → 最小检查”定位。
- **可演进**：代码更新时，你知道该检查哪些边界，而不是靠“感觉”修文档。

如果后续你希望把这套经验进一步产品化，最值得做的不是“再写更多文字”，而是把上面的检查清单做成脚本化的校验（例如对关键常量/术语做 CI 检查），让文档对齐变成自动化的一部分。
