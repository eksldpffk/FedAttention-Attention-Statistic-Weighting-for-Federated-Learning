import torch
from collections import defaultdict

class AttnStatsEMA:
    """Exponential moving average over simple attention stats per (layer, head).
    Keeps two stats per head: entropy, sparsity.
    """
    def __init__(self, beta=0.9, thresh=1e-3, device="cpu"):
        self.beta = beta
        self.thresh = thresh
        self.device = device
        self.state = defaultdict(lambda: None)  # (layer, head) -> tensor([entropy, sparsity])

    @torch.no_grad()
    def update_from_attentions(self, attentions):
        # attentions может быть None или содержать элементы None
        if attentions is None:
            return
        L = len(attentions)
        for l in range(L):
            A = attentions[l]
            if A is None:
                continue  # пропускаем слой без карт внимания
            # ожидаем форму (B, H, T, T)
            if A.ndim != 4:
                continue
            A = A.clamp_min(1e-12)
            # энтропия внимания и разреженность по каждой голове
            ent = -(A * A.log()).sum(dim=-1).mean(dim=(0, 2))                  # (H,)
            spars = (A < self.thresh).float().mean(dim=(-1, -2)).mean(dim=0)   # (H,)
            H = ent.shape[0]
            for h in range(H):
                key = (l, h)
                cur = torch.stack([ent[h], spars[h]]).to(self.device)
                prev = self.state[key]
                self.state[key] = cur if prev is None else self.beta*prev + (1-self.beta)*cur


    def pack_for_send(self):
        if not self.state:
            return []
        layers = sorted({l for (l, _) in self.state.keys()})
        per_layer = []
        for l in layers:
            heads = sorted([h for (ll, h) in self.state.keys() if ll == l])
            mat = torch.stack([self.state[(l, h)] for h in heads], dim=0)  # (H, 2)
            per_layer.append(mat.detach().cpu())
        return per_layer
