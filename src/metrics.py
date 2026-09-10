# src/metrics.py
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DownloadConfig


# ---------- HF robustness ----------
def safe_load_dataset(*args, retries: int = 3, sleep_s: float = 3.0, **kwargs):
    """
    Надёжная загрузка датасета (HuggingFace). Если сеть тупит — повторяем.
    Если всё равно не получилось — пробрасываем исключение (его поймает main и не уронит тренировку).
    """
    dl_cfg = kwargs.pop("download_config", None)
    if dl_cfg is None:
        dl_cfg = DownloadConfig(max_retries=10)
    kwargs["download_config"] = dl_cfg

    last_err = None
    for i in range(retries):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            last_err = e
            if i < retries - 1:
                time.sleep(sleep_s * (i + 1))
            else:
                raise last_err


def ppl_eval(
    model,
    eval_texts=None,
    model_name="distilgpt2",
    max_len=256,
    use_wikitext_val=True,
):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.eval()
    losses = []

    if use_wikitext_val:
        ds = safe_load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [
            x["text"]
            for x in ds.select(range(min(512, len(ds))))
            if isinstance(x["text"], str) and len(x["text"]) > 0
        ]
    else:
        texts = (eval_texts or [])[:256]

    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())

    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))


def _ppl_on_texts(model, tok, texts, max_len=256):
    model.eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            losses.append(out.loss.item())
    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))


def per_domain_ppl_eval(
    model,
    model_name: str = "distilgpt2",
    max_len: int = 256,
    max_texts_per_domain: int = 256,
):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # NEWS
    ag = safe_load_dataset("ag_news", split="train[4000:5000]")
    news_texts = [ex["text"] for ex in ag if isinstance(ex.get("text", None), str)]
    news_texts = news_texts[:max_texts_per_domain]

    # SCIENCE
    sci = safe_load_dataset("ccdv/arxiv-summarization", split="train[3000:3400]")
    sci_texts = [ex["article"] for ex in sci if isinstance(ex.get("article", None), str)]
    sci_texts = sci_texts[:max_texts_per_domain]

    # CODE (требует trust_remote_code=True)
    code_ds = safe_load_dataset(
        "code_search_net",
        "python",
        split="train[4000:4400]",
        trust_remote_code=True,
    )
    code_texts = []
    for ex in code_ds:
        if isinstance(ex.get("func_code_string", None), str) and ex["func_code_string"]:
            code_texts.append(ex["func_code_string"])
        elif isinstance(ex.get("code", None), str) and ex["code"]:
            code_texts.append(ex["code"])
    code_texts = code_texts[:max_texts_per_domain]

    ppls = {
        "news": _ppl_on_texts(model, tok, news_texts, max_len=max_len),
        "science": _ppl_on_texts(model, tok, sci_texts, max_len=max_len),
        "code": _ppl_on_texts(model, tok, code_texts, max_len=max_len),
    }
    return ppls


def attn_entropy_eval(
    global_model,
    model_name: str = "distilgpt2",
    max_len: int = 256,
    num_texts: int = 128,
):
    """
    ВАЖНО: создаём отдельную модель с attn_implementation="eager",
    грузим state_dict глобальной модели и безопасно считаем attention entropy.
    Это полностью убирает ошибку:
      "output_attentions is not supported when using attn_implementation=sdpa"
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = safe_load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [
        x["text"]
        for x in ds.select(range(min(num_texts, len(ds))))
        if isinstance(x["text"], str) and len(x["text"]) > 0
    ]

    device = global_model.device

    # отдельная модель для attention
    try:
        m = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        if hasattr(m.config, "attn_implementation"):
            m.config.attn_implementation = "eager"

    m.load_state_dict(global_model.state_dict(), strict=False)
    m.eval()
    m.config.output_attentions = True
    m.config.return_dict = True
    m.config.use_cache = False

    layer_sums = None
    layer_counts = None

    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = m(**enc, labels=enc["input_ids"], output_attentions=True)
            attns = out.attentions
            if attns is None:
                continue

            L = len(attns)
            if layer_sums is None:
                layer_sums = [0.0 for _ in range(L)]
                layer_counts = [0 for _ in range(L)]

            for l in range(L):
                A = attns[l]
                if A is None or A.ndim != 4:
                    continue
                # (B,H,T,T)
                A = A.clamp_min(1e-12)
                ent = -(A * A.log()).sum(dim=-1).mean(dim=(0, 2))  # (H,)
                ent_mean = float(ent.mean().item())
                layer_sums[l] += ent_mean
                layer_counts[l] += 1

    if layer_sums is None:
        return []

    mean_ents = []
    for s, c in zip(layer_sums, layer_counts):
        mean_ents.append(s / c if c > 0 else float("nan"))
    return mean_ents
