# FedAttention Attention Statistic Weighting for Federated Learning.
A federated learning method that uses Transformer attention statistics to guide how client updates are aggregated under non-IID data.

## Problem statement

**Federative learning** - this is a learning paradigm in which data remains locally with users (on devices), and only updates to model parameters are sent to the server. The server aggregates these updates and creates a new global model. <br>

<p>
  <img src="assets/FL_cf_str.png" align="right" width="350">
<br>
  FL is widely used when data is distributed among clients, but in reality, the clients:
  <ul>
    <li> have different data domains (code, news, scientific texts, etc.) </li>
    <li> don't learn the same way </li>
    <li> produce models of different quality </li>
  </ul>
<br>
  However, FedAvg averages everyone equally, which results in:
  <ul>
    <li> loss of useful information </li>
    <li> deterioration of quality </li>
    <li> instability with strong domain shift (non-IID) </li>
  </ul>
</p>


<br><br>
<p align="center">
  <i>In some cases, the client is useful, in others not, but FedAvg does not see this.</i>
</p>


## Propose approach

The idea is to use the model's internal signal - **self-attention**.
Explanation:
* Stable and confident models have focused and sparse attention (low entropy, high selectivity).
* Noisy/domain-mixed models are the opposite.

This means that the “quality” of a client can be determined based on the structure of its attention.
With this approach, the model is: 
- less sensitive to bad customers,
- more resilient,
- more adaptive

<p>
  <img src="assets/FL_wf.png" align="right" width="300">
  <br>

  Each client performs normal local training and also collects for each Transformer layer (l), attention head (h), and client (k) two signals are computed from the local attention statistics:
  <ul>
    <li><b>Attention entropy:</b> H = -Σ A log A</li>
    <li><b>Attention sparsity:</b> S = (1/N) Σ I(A ≤ ε)</li>
  </ul>

  The server combines these values into a per-layer score:
  <br><b>S<sub>k</sub><sup>l</sup> = mean<sub>h</sub>(τ<sub>e</sub> · (1 / Entropy<sub>l,h,k</sub>) + τ<sub>s</sub> · Sparsity<sub>l,h,k</sub> </b><br>
  And then converts the scores into layer-wise aggregation weights using softmax.

  Higher-weight client updates contribute more strongly to that attention layer, while the rest of the model is averaged normally.
</p>

## Results

<p align="center">
  <img src="assets/FL_results.png" widht="600">
</p>

1. **Global Performance:** FedAttention matches or slightly improves global PPL across all seeds.
2. **Domain Robustness:** FedAttention consistently improves worst-domain PPL (up to –10.8 points) and reduces variance across domains.
3. **Non-IID Stability:** FedAttention yields lower per-domain std, indicating more uniform generalization across heterogeneous clients.
4. **Strongest Effect Under Harder Conditions:** With more local steps (180) and more rounds (12), FedAttention shows substantial gains in all domain metrics.
5. **Structural Effects:** Attention entropy decreases across experiments → attention becomes more selective and confident, matching the theoretical motivation.

_FedAttention demonstrates a robust, interpretable, and practically meaningful improvement over FedAvg, especially in non-IID settings._




