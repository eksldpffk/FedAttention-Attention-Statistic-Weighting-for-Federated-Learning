from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from .attn_stats import AttnStatsEMA

def _collate(batch, pad_id: int):
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=int(pad_id))
    attn = (ids != int(pad_id)).to(torch.long)
    return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import torch
from .attn_stats import AttnStatsEMA

def _collate(batch, pad_id: int):
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=int(pad_id))
    attn = (ids != int(pad_id)).to(torch.long)
    return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}

def client_train_step(
    model_name,
    shard,
    lr=3e-5,
    local_steps=200,
    batch_size=8,
    device="cuda",
    init_state=None,
    prox_mu: float = 0.0
):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # модель с поддержкой карт внимания
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager"
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    model.config.output_attentions = True
    model.config.return_dict = True
    model.config.use_cache = False

    # заливаем глобальные веса
    if init_state is not None:
        model.load_state_dict(init_state, strict=False)

    model.train()

    # подготовим global params на девайсе один раз (для FedProx)
    global_params = None
    if prox_mu is not None and prox_mu > 0.0 and init_state is not None:
        global_params = {k: v.to(device) for k, v in init_state.items()}

    # DataLoader
    dl = DataLoader(
        shard["data"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate(b, tok.pad_token_id),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, 0, local_steps)
    stat_ema = AttnStatsEMA(beta=0.9, thresh=1e-3, device=device)

    it, total_loss = 0, 0.0
    for batch in dl:
        for k in batch:
            batch[k] = batch[k].to(device)

        out = model(**batch, output_attentions=True)
        loss = out.loss

        prox_term = None
        if global_params is not None and prox_mu > 0.0:
            prox = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad and name in global_params:
                    prox = prox + (p - global_params[name]).pow(2).sum()
            prox_term = 0.5 * prox_mu * prox
            loss = loss + prox_term

        # dbg – один раз проверим
        if it == 5:
            print("dbg: prox_mu =", prox_mu, " global_params is None? ", (global_params is None))
            attns = out.attentions
            print("dbg: attentions is None? ", attns is None)
            if attns is not None:
                first = next((a for a in attns if a is not None), None)
                shp = tuple(first.shape) if first is not None else None
                print("dbg: num layers =", len(attns), " first non-None shape =", shp)

            if prox_term is not None:
                print("dbg: prox term =", float(prox_term.detach().cpu()))


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        scheduler.step()

        stat_ema.update_from_attentions(out.attentions)

        total_loss += float(loss.item())
        it += 1
        if it >= local_steps:
            break

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    stats = stat_ema.pack_for_send()
    print("attn layers:", len(stats), " first layer heads:", (stats[0].shape[0] if stats else 0))
    return state, stats, (total_loss / max(1, it))

