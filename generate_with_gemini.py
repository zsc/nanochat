#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TEMPLATE_PATH = Path("template.md")


DEFAULT_TARGETS = [
    "chapter4.md",
    "chapter5.md",
    "chapter7.md",
    "chapter8.md",
    "chapter9.md",
    "chapter10.md",
    "chapter11.md",
    "chapter12.md",
    "chapter13.md",
    "chapter14.md",
    "appendixA.md",
    "appendixB.md",
    "appendixC.md",
    "appendixD.md",
]


API_KEY_ENV_CANDIDATES = [
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_API_KEY",
    "GENAI_API_KEY",
]


def _get_api_key() -> str:
    for key in API_KEY_ENV_CANDIDATES:
        value = os.environ.get(key)
        if value:
            return value
    raise SystemExit(
        "Missing Gemini API key. Set one of: "
        + ", ".join(API_KEY_ENV_CANDIDATES)
        + "."
    )


def _normalize_model_name(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _strip_wrapping_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t

    first_newline = t.find("\n")
    if first_newline == -1:
        return t

    fence_line = t[:first_newline].strip()
    if not fence_line.startswith("```"):
        return t

    if not t.endswith("```"):
        return t

    inner = t[first_newline + 1 : -3].strip()
    return inner


def _bytes_len_utf8(text: str) -> int:
    return len(text.encode("utf-8"))


def _extract_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text
    return ""


def _send_message_with_retry(chat, message: str, *, attempts: int = 4, base_sleep_s: float = 1.5):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = chat.send_message(message)
            text = _extract_text(resp)
            if text and text.strip():
                return resp
            raise RuntimeError("Empty response text")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            print(
                f"[retry] send_message failed (attempt {attempt}/{attempts}): {type(exc).__name__}: {exc}. sleeping {sleep_s:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


@dataclass(frozen=True)
class GenConfig:
    model: str
    temperature: float
    max_output_tokens: int
    thinking_budget: Optional[int]
    min_bytes: int
    max_passes: int
    sleep_s: float


def _build_base_prompt(template: str, target_filename: str) -> str:
    prompt = template

    # User requested: replace the `chapter1.md` name in the "file organization" sentence.
    prompt = prompt.replace(
        "文件组织是 index.md + chapter1.md + ...",
        f"文件组织是 index.md + {target_filename} + ...",
    )

    # Replace the tail instruction line (template currently contains one).
    prompt = re.sub(
        r"高质量生成\s+\S+\.md。尽量详细。\s*$",
        f"高质量生成 {target_filename}。尽量详细。",
        prompt.strip(),
        flags=re.MULTILINE,
    )

    prompt += (
        "\n\n---\n"
        f"目标文件：`{target_filename}`\n"
        "输出要求：\n"
        "- 只输出该文件的 Markdown 正文，不要解释、不要前后包裹、不要用占位符或省略号。\n"
        "- 全中文叙述为主；不写程序代码；允许公式（LaTeX）与 ASCII 图。\n"
        "- 所有数学表达式都用 LaTeX：行内用 `$...$`，独立成行用 `$$...$$`；不要用反引号 `...` 表示公式，也不要用 `\\(\\)`/`\\[\\]`。\n"
        "- 严格遵守“章节结构要求”（开篇段落/文字论述/本章小结/练习题/常见陷阱与错误）。\n"
        "- 练习题：6-8 题；每题包含 Hint；答案用 <details> 折叠；答案不包含代码。\n"
        "- 章节编号与小节编号要自洽（如 10.1、10.2… 或 A.1、A.2…）。\n"
    )
    return prompt


def _build_expand_prompt(target_filename: str, min_bytes: int, current_bytes: int | None) -> str:
    current_hint = "" if current_bytes is None else f"（当前约 {current_bytes} 字节）"
    return (
        "方向正确，大幅扩写。\n"
        f"请在保持整体结构与章节编号不变的前提下，把论述深度、例子、公式直觉、ASCII 图、rule-of-thumb、练习题 Hint/折叠答案、Gotchas 都显著扩充；"
        "把每个小节写成“能独立读懂并可复习”的密度。\n"
        "所有数学表达式都用 LaTeX：行内用 `$...$`，独立成行用 `$$...$$`。\n"
        f"请输出完整的 `{target_filename}`（不要只输出增量），并确保 UTF-8 文件大小 ≥ {min_bytes} 字节{current_hint}；不要使用占位符或省略号；不要输出任何生成说明。\n"
    )


def _build_extra_expand_prompt(target_filename: str, min_bytes: int, current_bytes: int) -> str:
    return (
        "继续扩写，重点补足仍然偏短/偏浅的部分（尤其是公式直觉、类比、Gotchas 与练习题答案的解释深度）。\n"
        "所有数学表达式都用 LaTeX：行内用 `$...$`，独立成行用 `$$...$$`。\n"
        f"请输出完整的 `{target_filename}`，并确保 UTF-8 文件大小 ≥ {min_bytes} 字节（当前约 {current_bytes} 字节）。\n"
        "不要输出任何生成说明。\n"
    )


def _make_genai_client(api_key: str):
    from google import genai  # type: ignore

    # Prefer the official/simple path: GEMINI_API_KEY in env (matches docs/examples).
    if os.environ.get("GEMINI_API_KEY"):
        return genai.Client()
    return genai.Client(api_key=api_key)


def _model_supports_thinking(client, model: str) -> bool:
    try:
        info = client.models.get(model=model)
        return bool(getattr(info, "thinking", False))
    except Exception:
        return False


def _make_generate_config(client, cfg: GenConfig):
    from google.genai import types  # type: ignore

    config = types.GenerateContentConfig(
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        response_mime_type="text/plain",
        system_instruction=(
            "你是中文教材作者。输出必须是可直接保存为 Markdown 文件的正文；不输出解释。"
        ),
    )

    if cfg.thinking_budget is None:
        return config

    if _model_supports_thinking(client, cfg.model):
        config.thinking_config = types.ThinkingConfig(thinking_budget=cfg.thinking_budget)
    return config


def _generate_one_file(
    *,
    client,
    cfg: GenConfig,
    template: str,
    target_path: Path,
    overwrite: bool,
) -> None:
    if target_path.exists() and not overwrite:
        print(
            f"[skip] {target_path} (exists; use --overwrite to replace)", flush=True
        )
        return

    gen_config = _make_generate_config(client, cfg)

    chat = client.chats.create(model=cfg.model, config=gen_config)

    base_prompt = _build_base_prompt(template, target_path.name)
    print(f"[1/..] {target_path.name}: generating draft...", flush=True)
    resp1 = _send_message_with_retry(chat, base_prompt)
    text1 = _strip_wrapping_fence(_extract_text(resp1)).rstrip() + "\n"

    # Expansion pass (must be in the same chat session).
    print(f"[2/..] {target_path.name}: expanding (same session)...", flush=True)
    resp2 = _send_message_with_retry(
        chat,
        _build_expand_prompt(
            target_filename=target_path.name,
            min_bytes=cfg.min_bytes,
            current_bytes=_bytes_len_utf8(text1),
        )
    )
    text2 = _strip_wrapping_fence(_extract_text(resp2)).rstrip() + "\n"

    best_text = text2 if _bytes_len_utf8(text2) >= _bytes_len_utf8(text1) else text1
    best_bytes = _bytes_len_utf8(best_text)

    passes_used = 2
    while best_bytes < cfg.min_bytes and passes_used < cfg.max_passes:
        passes_used += 1
        print(
            f"[{passes_used}/..] {target_path.name}: still short ({best_bytes} bytes), expanding again...",
            flush=True,
        )
        resp_extra = _send_message_with_retry(
            chat,
            _build_extra_expand_prompt(
                target_filename=target_path.name,
                min_bytes=cfg.min_bytes,
                current_bytes=best_bytes,
            )
        )
        candidate_text = _strip_wrapping_fence(_extract_text(resp_extra)).rstrip() + "\n"
        candidate_bytes = _bytes_len_utf8(candidate_text)
        if candidate_bytes >= best_bytes:
            best_text = candidate_text
            best_bytes = candidate_bytes
        if cfg.sleep_s > 0:
            time.sleep(cfg.sleep_s)

    _write_text_atomic(target_path, best_text)
    status = "ok" if best_bytes >= cfg.min_bytes else "warn"
    print(f"[{status}] wrote {target_path.name} ({best_bytes} bytes)", flush=True)

    if cfg.sleep_s > 0:
        time.sleep(cfg.sleep_s)


def _iter_targets(args_targets: list[str]) -> list[str]:
    if args_targets:
        return args_targets
    return DEFAULT_TARGETS


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate chapters/appendices from template.md using Gemini (API key from env)."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Output filenames to generate (default: chapter4,5,7-14 + appendixA-D).",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Gemini model name (default: gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=20000,
        help="Max output tokens per request (default: 20000).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=4096,
        help="Thinking budget for thinking models (default: 4096). Ignored if model doesn't support thinking.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=23000,
        help="Minimum UTF-8 bytes for final file (default: 23000).",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Max total passes per file (default: 3; initial + expansion + optional extra).",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.25,
        help="Sleep seconds between API calls (default: 0.25).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing files (default: true).",
    )

    args = parser.parse_args(argv)

    if args.max_passes < 2:
        raise SystemExit("--max-passes must be >= 2 (needs draft + expansion).")

    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Missing {TEMPLATE_PATH}. Run from repo root.")

    api_key = _get_api_key()

    model = _normalize_model_name(args.model)

    cfg = GenConfig(
        model=model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
        min_bytes=args.min_bytes,
        max_passes=args.max_passes,
        sleep_s=args.sleep_s,
    )

    template = _read_text(TEMPLATE_PATH)
    client = _make_genai_client(api_key)

    targets = _iter_targets(args.targets)
    for name in targets:
        target_path = Path(name)
        _generate_one_file(
            client=client,
            cfg=cfg,
            template=template,
            target_path=target_path,
            overwrite=args.overwrite,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
