from __future__ import annotations

import copy
from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM

_FEDATTN_EMA: Dict[str, torch.Tensor] = {} 


def init_global(model_name: str, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model


def fedavg(models_sd):
    keys = models_sd[0].keys()
    agg = {k: sum(sd[k] for sd in models_sd) / len(models_sd) for k in keys}
    return agg


def _layer_index_from_key(k: str) -> Optional[int]:
    if k.startswith("transformer.h.") and ".attn." in k:
        try:
            return int(k.split(".")[2])
        except Exception:
            return None
    return None


def fedattention(
    models_sd,
    attn_stats_list,
    reference_model,
    temperature: float | None = 0.10,
    tau_e: float = 1.7,
    tau_s: float = 0.8,
    min_alpha_frac: float = 0.04,
    round_idx: int = 1,
    cap: float = 0.60,
    ema_alpha: float = 0.7,
):
    global _FEDATTN_EMA

    nonempty = [S for S in attn_stats_list if isinstance(S, list) and len(S) > 0]
    if not nonempty:
        return fedavg(models_sd)

    L = min(len(S) for S in nonempty)
    if L <= 0:
        return fedavg(models_sd)

    layer_weights = []
    K = len(models_sd)

    for l in range(L):
        per_client = [S[l] for S in nonempty if len(S) > l]
        if not per_client:
            layer_weights.append(torch.ones(K) / K)
            continue

        ents = torch.stack([M[:, 0] for M in per_client], dim=0)
        spars = torch.stack([M[:, 1] for M in per_client], dim=0)

        eps = 1e-6
        inv_ent = 1.0 / (ents + eps)
        score = tau_e * inv_ent + tau_s * spars
        score = score.mean(dim=1)

        center = (l + 0.5) / max(1, L)
        depth_gain = 1.0 + 0.5 * (1.0 - abs(2 * center - 1.0))
        score = score * depth_gain

        if temperature is None:
            t0, t1 = 0.12, 0.20
            lam = min(1.0, max(0.0, (round_idx - 1) / 2.0))
            temp = t0 * (1 - lam) + t1 * lam
        else:
            temp = float(temperature)

        w_sub = torch.softmax(score / max(temp, 1e-8), dim=0)  # (K',)

        if cap is not None and cap > 0:
            over = w_sub > cap
            rest = (~over).sum().item()
            if over.any().item() and rest > 0:
                extra = (w_sub[over] - cap).sum()
                w_sub[over] = cap
                w_sub[~over] += extra / rest

        min_a = min_alpha_frac / max(1, len(w_sub))
        w_sub = torch.clamp(w_sub, min=min_a)
        w_sub = w_sub / w_sub.sum()

        w_full = torch.zeros(K)
        for i in range(min(K, len(w_sub))):
            w_full[i] = w_sub[i]
        if w_full.sum() <= 0:
            w_full[:] = 1.0 / K
        else:
            w_full = w_full / w_full.sum()

        ema_key = f"layer_{l}_K{K}"
        if ema_key in _FEDATTN_EMA:
            w_full = ema_alpha * _FEDATTN_EMA[ema_key] + (1 - ema_alpha) * w_full
            w_full = w_full / w_full.sum()
        _FEDATTN_EMA[ema_key] = w_full.detach().clone()

        layer_weights.append(w_full)

    if layer_weights:
        w0 = layer_weights[0]
        print(f"[FedAttn] L0 weights: mean={w0.mean():.3f} std={w0.std():.3f} -> {w0.tolist()}")

    agg = copy.deepcopy(reference_model.state_dict())
    keys = list(agg.keys())
    for k in keys:
        ll = _layer_index_from_key(k)
        if (ll is None) or (ll < 0) or (ll >= L):
            agg[k] = sum(sd[k] for sd in models_sd) / K
        else:
            w = layer_weights[ll]  # (K,)
            stacked = torch.stack([sd[k] for sd in models_sd], dim=0)
            agg[k] = (w.view(-1, *([1] * (stacked.dim() - 1))) * stacked).sum(dim=0)

    return agg


def get_fedattn_ema_state() -> Dict[str, Any]:
    out = {}
    for k, v in _FEDATTN_EMA.items():
        out[k] = v.detach().cpu()
    return out


def set_fedattn_ema_state(state: Dict[str, Any] | None):
    global _FEDATTN_EMA
    _FEDATTN_EMA = {}
    if not state:
        return
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            _FEDATTN_EMA[k] = v.detach().cpu()
