# src/main.py
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, Optional, Tuple

import torch

from .data_prep import load_domains, tokenize_domains, make_clients
from .client import client_train_step
from .server import init_global, fedavg, fedattention, get_fedattn_ema_state, set_fedattn_ema_state
from .metrics import ppl_eval, per_domain_ppl_eval, attn_entropy_eval

MODEL = "distilgpt2"
CHECKPOINT_DIR = "checkpoints"

EVAL_SNIPPETS = [
    "This is a test about science and mathematics.",
    "for (int i=0;i<10;i++) { cout<<i; }",
    "The results indicate that",
    "In this paper, we propose",
] * 64


def _compute_and_print_metrics(global_model):
    # Если сеть умерла — НЕ роняем блять процесс, заебалась уже
    try:
        ppl = ppl_eval(global_model, EVAL_SNIPPETS, model_name=MODEL)
        print(f"[Eval] Global Val PPL={ppl:.2f}")
    except Exception as e:
        print(f"[Eval] Global Val PPL failed (network/caching issue): {repr(e)}")
        ppl = None

    try:
        domain_ppls = per_domain_ppl_eval(global_model, model_name=MODEL)
        vals = [v for v in domain_ppls.values() if isinstance(v, float)]
        if vals:
            mean_ppl = sum(vals) / len(vals)
            worst_ppl = max(vals)
            var = sum((x - mean_ppl) ** 2 for x in vals) / len(vals)
            std = var ** 0.5
            print(f"[Per-domain PPL] {domain_ppls}")
            print(f"[Per-domain Summary] mean={mean_ppl:.2f}  worst={worst_ppl:.2f}  std={std:.2f}")
        else:
            print("[Per-domain PPL] no valid values")
    except Exception as e:
        print(f"[Per-domain PPL] failed (network/caching issue): {repr(e)}")

    try:
        layer_ents = attn_entropy_eval(global_model, model_name=MODEL)
        if layer_ents:
            mean_ent = sum(layer_ents) / len(layer_ents)
            pretty = ", ".join(f"{e:.3f}" for e in layer_ents)
            print(f"[AttnEntropy] per-layer: [{pretty}]  (mean={mean_ent:.3f})")
        else:
            print("[AttnEntropy] could not compute (no attentions)")
    except Exception as e:
        print(f"[AttnEntropy] failed: {repr(e)}")


def _state_paths(strategy: str, seed: int) -> Tuple[str, str]:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, f"{MODEL}_{strategy}_seed{seed}_best.pt")
    last_path = os.path.join(CHECKPOINT_DIR, f"{MODEL}_{strategy}_seed{seed}_last.pt")
    return best_path, last_path


def _save_last_state(path: str, payload: Dict[str, Any]):
    torch.save(payload, path)


def _load_last_state(path: str, map_location: str):
    return torch.load(path, map_location=map_location)


def run(
    rounds: int = 3,
    clients_per_round: int = 4,
    local_steps: int = 100,
    strategy: str = "fedattention",
    device: str = "auto",
    seed: int = 42,
    warmup_rounds: int = 1,
    prox_mu: float = 0.0,
    resume: bool = False,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # FedAvg/FedProx не нужен warmup
    if strategy.lower() in ("fedavg", "fedprox"):
        warmup_rounds = 0

    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    domains = load_domains()
    tokenized = tokenize_domains(domains)
    clients = make_clients(
        tokenized,
        clients_per_domain=max(2, clients_per_round),
        samples_per_client=400,
        seed=seed,
    )
    random.Random(seed).shuffle(clients)

    global_model = init_global(MODEL, device=device)

    best_path, last_path = _state_paths(strategy, seed)

    start_round = 1
    best_ppl = 1e9
    best_sd = None

    if resume and os.path.exists(last_path):
        st = _load_last_state(last_path, map_location=device)
        global_model.load_state_dict(st["global_sd"], strict=False)
        start_round = int(st.get("last_round", 0)) + 1

        best_ppl = float(st.get("best_ppl", 1e9))
        best_sd = st.get("best_sd", None)

        if strategy.lower() == "fedattention":
            set_fedattn_ema_state(st.get("fedattn_ema", None))

        if "py_random_state" in st:
            random.setstate(st["py_random_state"])
        if "torch_random_state" in st:
            torch.random.set_rng_state(st["torch_random_state"])

        print(f"[Resume] loaded {last_path}. Continue from round {start_round}/{rounds}")
    else:
        if strategy.lower() == "fedattention":
            set_fedattn_ema_state(None)

    for r in range(start_round, rounds + 1):
        start = (r - 1) * clients_per_round % len(clients)
        picked = clients[start:start + clients_per_round]

        global_sd_cpu = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

        client_states, client_stats, losses = [], [], []
        for c in picked:
            state, stats, loss = client_train_step(
                MODEL,
                c,
                local_steps=local_steps,
                device=device,
                init_state=global_sd_cpu,
                prox_mu=(prox_mu if strategy.lower() == "fedprox" else 0.0),
            )
            client_states.append(state)
            client_stats.append(stats)
            losses.append(loss)

        # aggregate
        if strategy.lower() in ("fedavg", "fedprox") or r <= warmup_rounds:
            new_sd = fedavg(client_states)
        else:
            new_sd = fedattention(client_states, client_stats, global_model, round_idx=r)

        # update global with momentum blending (here is mu, but in the readme its alpha on the last stage in aproache's discription)
        mu_blend = 0.3
        old = global_model.state_dict()
        for k in new_sd:
            new_sd[k] = mu_blend * old[k] + (1 - mu_blend) * new_sd[k]
        global_model.load_state_dict(new_sd, strict=False)

        # round ppl (может упасть при плохом интернете - тогда не роняем раунд)
        try:
            ppl = ppl_eval(global_model, EVAL_SNIPPETS, model_name=MODEL)
        except Exception as e:
            print(f"[Round {r:02d}] Val PPL failed (network/caching issue): {repr(e)}")
            ppl = float("inf")

        mean_loss = sum(losses) / max(1, len(losses))
        print(f"[Round {r:02d}] strategy={strategy}  mean client loss={mean_loss:.3f}  Val PPL={ppl:.2f}")

        if ppl < best_ppl:
            best_ppl = float(ppl)
            best_sd = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
            torch.save(best_sd, best_path)
            print(f"[Best] updated at round {r:02d}: PPL={best_ppl:.2f} saved -> {best_path}")

        payload = {
            "last_round": r,
            "global_sd": {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()},
            "best_ppl": best_ppl,
            "best_sd": best_sd,
            "py_random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        if strategy.lower() == "fedattention":
            payload["fedattn_ema"] = get_fedattn_ema_state()

        _save_last_state(last_path, payload)

    if os.path.exists(best_path):
        sd = torch.load(best_path, map_location=device)
        global_model.load_state_dict(sd, strict=False)
        print("[Best] restored best checkpoint from disk")

    _compute_and_print_metrics(global_model)
    print(f"[Run finished] best={best_path} last={last_path}")
    return global_model


def eval_only(strategy: str, device: str, seed: int):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    best_path, _ = _state_paths(strategy, seed)
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_path}. Run training first.")
    print(f"[Eval-only] Loading {best_path}")
    global_model = init_global(MODEL, device=device)
    sd = torch.load(best_path, map_location=device)
    global_model.load_state_dict(sd, strict=False)
    _compute_and_print_metrics(global_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients-per-round", type=int, default=4)
    ap.add_argument("--local-steps", type=int, default=100)
    ap.add_argument("--strategy", type=str, default="fedattention", choices=["fedavg", "fedattention", "fedprox"])
    ap.add_argument("--prox-mu", type=float, default=0.001)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-rounds", type=int, default=1)

    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint (last.pt) if exists")

    args = ap.parse_args()

    if args.eval_only:
        eval_only(strategy=args.strategy, device=args.device, seed=args.seed)
        return

    run(
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_steps=args.local_steps,
        strategy=args.strategy,
        device=args.device,
        seed=args.seed,
        warmup_rounds=args.warmup_rounds,
        prox_mu=args.prox_mu,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

