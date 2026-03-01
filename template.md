## 项目背景
面向模型训练和数据准备人员，生成一个 nanochat 的中文 markdown。

文件组织是 index.md + chapter1.md + ...
不写代码。
提供 rule-of-thumb。

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **文字论述**：以文字论述为主，适当配上ASCII 图说明。
3. **本章小结**：总结关键概念和公式
4. **练习题**：
   - 每章包含6-8道练习题
   - 50%基础题（帮助熟悉材料）
   - 50%挑战题（包括开放性思考题）
   - 每题提供提示（Hint）
   - 答案默认折叠，不包含代码
5. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧

---
# nanochat：面向模型训练与数据准备人员的中文手册

*对齐仓库版本：`168b28e`（更新日期：2026-03-01）*

本书面向“要把一个小模型从零训练到可对话”的一线同学：既关心**数据怎么来、怎么切、怎么喂**，也关心**训练怎么跑得快、评估怎么对齐、产物怎么落盘并能聊起来**。nanochat 的目标是把 Tokenizer → 预训练 → SFT → RL → 评估 → 推理/Chat UI 这条链路压到最小可读、最小可改、单机单节点可复现。

## 你会得到什么

- 一条可复现的端到端流水线（以 `runs/speedrun.sh` 为“事实标准”）。
- 以数据准备/训练工程视角，拆解每一阶段的**输入/输出约定**与**关键决策点**。
- 每章都提供可操作的经验法则（rule-of-thumb）、练习题与常见坑（Gotchas），用于上手与排障。

## 读者定位（建议你属于其中之一）

- 模型训练工程师：关注吞吐、显存、数值稳定、分布式细节与跑通全流程。
- 数据准备/数据工程：关注数据格式、切分、shuffle、packing、mask、数据混配与质量控制。
- 研究/实验同学：关注 scaling sweep、指标（BPB/CORE/ChatCORE）、实验记录与可比较性。

## 仓库导航（先认路再动手）

- `runs/`：端到端脚本入口（推荐从 `runs/speedrun.sh` 开始）。
- `scripts/`：各阶段的“可执行入口”（tok/base/sft/rl/eval/web）。
- `nanochat/`：核心库（模型、tokenizer、dataloader、engine、eval、checkpoint、report）。
- `tasks/`：SFT/评测数据集封装（多为 HuggingFace datasets + 统一对话格式）。
- `dev/`：参考/分析/文档（数据重打包参考、leaderboard、实验记录与图表）。

## 一张图看全流程（你将反复回到这张图）

```
预训练语料（parquet shards）          评测 bundle / 下游任务
          │                               │
          ▼                               ▼
Tokenizer 训练与评估  ─────────────►  BPB / CORE / 采样
 (scripts/tok_*)                      (scripts/base_eval.py)
          │
          ▼
Base 预训练（GPT） ────────────────►  保存 checkpoint + report
 (scripts/base_train.py)               (nanochat/checkpoint_manager.py, nanochat/report.py)
          │
          ▼
Chat SFT（对话/工具/多选能力） ─────►  Chat 评测（ChatCORE）
 (scripts/chat_sft.py)                 (scripts/chat_eval.py)
          │
          ▼
RL（GSM8K 上的采样 + reward） ──────►  Chat 推理/交互
 (scripts/chat_rl.py)                  (scripts/chat_cli.py / scripts/chat_web.py)
```

## 强约定（非常重要）

- **产物默认落盘到缓存目录**：默认 `~/.cache/nanochat`，可用环境变量 `NANOCHAT_BASE_DIR` 覆盖（见 `nanochat/common.py` 与 `runs/*.sh`）。
- **很多步骤会联网下载**：预训练数据（HuggingFace dataset）、CORE 评测包（S3）、identity conversations（S3）、部分合成数据脚本可能调用第三方 API（见 `dev/gen_synthetic_data.py`）。
- **本书以“可跑通与可解释”为第一优先级**：先把输入/输出、数据格式、指标与排障路径弄清楚，再谈花式技巧。

## 推荐阅读路线（按你的工作重心选）

- **我想最快跑通全流程**：第 1 章 → 第 2 章 → 第 3/4 章（数据+tokenizer）→ 第 5 章（预训练）→ 第 8 章（SFT）→ 第 10 章（推理/UI）。
- **我主要做数据准备**：第 3 章 → 第 4 章 → 第 6 章（packing/mask）→ 第 8 章（SFT 混配与格式）→ 第 7 章（指标如何被数据影响）。
- **我主要做训练/性能**：第 2 章 → 第 5 章 → 第 6 章 → 第 7 章 → 第 11 章（miniseries/scaling）。

## 目录（index + chapter1.md + ...）

> 下面链接的章节文件将逐步补齐。每章默认包含：开篇段落、文字论述（可含 ASCII 图）、本章小结、练习题（含 Hint，答案折叠且不含代码）、Gotchas。

1. [第 1 章：nanochat 是什么，以及它的“最小全流程”](chapter1.md)  
   目标：建立整体心智模型；知道“你在训练什么、数据从哪来、如何评估”。参考：`README.md`、`runs/speedrun.sh`。
2. [第 2 章：环境与可复现（单机多卡、缓存目录、依赖、报告）](chapter2.md)  
   目标：把一次实验跑成“可复现的产物”；理解 `NANOCHAT_BASE_DIR`、report、wandb 的边界。参考：`nanochat/common.py`、`nanochat/report.py`、`runs/*.sh`。
3. [第 3 章：预训练数据（FineWeb-Edu shards：下载、切分、shuffle、parquet 约定）](chapter3.md)  
   目标：理解 nanochat 预训练语料的组织方式与下载策略；知道“训练/验证”如何划分。参考：`nanochat/dataset.py`、`dev/repackage_data_reference.py`。
4. [第 4 章：Tokenizer（GPT-4 风格 BPE、特殊 token、压缩率与 BPB）](chapter4.md)  
   目标：从数据与训练视角理解 tokenizer：split pattern、special tokens、token_bytes 与 bpb。参考：`nanochat/tokenizer.py`、`scripts/tok_train.py`、`scripts/tok_eval.py`。
5. [第 5 章：预训练配方（一个旋钮：`depth`，其余自动计算）](chapter5.md)  
   目标：读懂 `base_train` 的“计算最优”思路：模型形状、批大小、训练步数（FLOPs 或 tokens:params）。参考：`scripts/base_train.py`、`nanochat/gpt.py`、`nanochat/optim.py`。
6. [第 6 章：喂数这件事（DDP 分片、BOS 对齐、best-fit packing、恢复训练）](chapter6.md)  
   目标：把“文本→tokens→(B,T) batch”这条链路讲透；知道裁剪/填充/利用率的权衡。参考：`nanochat/dataloader.py`、`nanochat/tokenizer.py`（conversation render/mask）。
7. [第 7 章：评估与指标（BPB、CORE、采样；如何读懂 leaderboard）](chapter7.md)  
   目标：把“训练信号”和“能力信号”分开看；知道何时用 BPB、何时用 CORE/ChatCORE。参考：`scripts/base_eval.py`、`nanochat/core_eval.py`、`dev/LEADERBOARD.md`。
8. [第 8 章：SFT 数据准备（对话格式、mask、任务混合、identity 合成）](chapter8.md)  
   目标：把 SFT 训练数据变成可控的工程系统：格式、mixing、oversample、数据质量。参考：`scripts/chat_sft.py`、`tasks/`、`dev/gen_synthetic_data.py`、`tasks/customjson.py`。
9. [第 9 章：RL（GSM8K 上的采样、reward、优势归一化与稳定性）](chapter9.md)  
   目标：理解 nanochat 的 RL 版本（类 GRPO/REINFORCE）到底做了什么、没做什么，以及它依赖的数据/评估约定。参考：`scripts/chat_rl.py`、`tasks/gsm8k.py`、`nanochat/engine.py`。
10. [第 10 章：推理与交互（Engine/KV cache、工具调用、CLI/Web UI）](chapter10.md)  
   目标：把“能训”变成“能聊”；理解推理缓存、批量采样与工具 token 的状态机。参考：`nanochat/engine.py`、`scripts/chat_cli.py`、`scripts/chat_web.py`、`nanochat/ui.html`。
11. [第 11 章：实验体系（miniseries/scaling laws：算力预算、结果记录与分析）](chapter11.md)  
   目标：把实验做成“可比较的数据集”；理解 sweep 脚本如何组织实验与提取指标。参考：`runs/miniseries.sh`、`runs/scaling_laws.sh`、`dev/scaling_analysis.ipynb`。
12. [第 12 章：排障与最佳实践（数据质量×吞吐×数值稳定的交叉坑位）](chapter12.md)  
   目标：把常见问题收敛成 checklist：OOM、慢、发散、指标抖、数据坏、恢复难。参考：贯穿全仓库（重点：`scripts/*`、`nanochat/dataloader.py`、`nanochat/flash_attention.py`、`nanochat/fp8.py`）。

## 全书通用的 rule-of-thumb（先记住这几条）

- 以 `runs/speedrun.sh` 作为“端到端正确性”的基线；任何局部优化都要回到这条链路验证。
- 先用小深度（例如 `d12`）把数据/训练/评估跑顺，再上 `d24/d26` 追能力与速度。
- OOM/不稳时，优先动 **每卡 batch**（`--device-batch-size`），再动 **总 batch**（`--total-batch-size`），最后才动模型与序列长度。
- 数据准备的第一性：**格式一致、可复现的 shuffle、可解释的 packing/mask**；否则训练细节再漂亮也会被数据噪声淹没。

---
高质量生成 chapter1.md。尽量详细。
