import random
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "distilgpt2"

def load_domains(seed: int = 42,
                 max_per_domain: int = 1500):
    """
    Загружает тексты из трёх очень разных доменов:
    - news: новости (ag_news)
    - science: научные абстракты (ccdv/arxiv-summarization)
    - code: код Python (code_search_net)
    Возвращает dict: {domain_name: [str, str, ...]}.
    """
    random.seed(seed)

    domains = {}

    # Новости
    ag = load_dataset("ag_news", split="train[:4000]", trust_remote_code=True)
    news_texts = [ex["text"] for ex in ag]
    random.shuffle(news_texts)
    domains["news"] = news_texts[:max_per_domain]

    # Научные статьи: arXiv
    # берём датасет ccdv/arxiv-summarization, поле "article" = полный текст статьи
    sci = load_dataset("ccdv/arxiv-summarization", split="train[:3000]", trust_remote_code=True)
    sci_texts = [ex["article"] for ex in sci if ex.get("article")]
    random.shuffle(sci_texts)
    domains["science"] = sci_texts[:max_per_domain]

    # Код (Python)
    code_ds = load_dataset("code_search_net", "python", split="train[:4000]", trust_remote_code=True)
    # у разных версий датасета ключи могут называться по-разному
    # пробуем несколько вариантов
    code_texts = []
    for ex in code_ds:
        if "func_code_string" in ex and ex["func_code_string"]:
            code_texts.append(ex["func_code_string"])
        elif "code" in ex and ex["code"]:
            code_texts.append(ex["code"])
    random.shuffle(code_texts)
    domains["code"] = code_texts[:max_per_domain]

    print({k: len(v) for k, v in domains.items()})
    return domains


def tokenize_domains(domains, model_name: str = MODEL, max_length: int = 256):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tokenized = {}
    for name, texts in domains.items():
        enc = tok(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        # enc["input_ids"] - список списков id
        tokenized[name] = enc["input_ids"]
        print(f"[tokenize_domains] {name}: {len(tokenized[name])} sequences")
    return tokenized


def make_clients(tokenized_domains,
                 clients_per_domain: int = 3,
                 samples_per_client: int = 400,
                 seed: int = 42):
    random.seed(seed)
    clients = []
    cid = 0

    for dom_name, seqs in tokenized_domains.items():
        seqs = list(seqs)
        random.shuffle(seqs)

        needed = clients_per_domain * samples_per_client
        if len(seqs) < needed:
            # если данных мало - просто уменьшаем samples_per_client
            samples_per_client = max(1, len(seqs) // clients_per_domain)

        for i in range(clients_per_domain):
            start = i * samples_per_client
            end = start + samples_per_client
            shard = seqs[start:end]
            if not shard:
                continue

            data = [{"input_ids": s} for s in shard]
            clients.append({
                "id": f"{dom_name}_c{i}",
                "domain": dom_name,
                "data": data,
            })
            cid += 1

    print(f"[make_clients] total clients: {len(clients)}")
    return clients

